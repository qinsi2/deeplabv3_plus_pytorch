from turtle import forward
from numpy import block
import torch
from torch import nn 
import math
import os
import torch.utils.model_zoo as model_zoo

# mobilenetv2的结构图详见本文件夹下的图片

def conv_bn(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(in_c * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_c == out_c

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #----------------
                # 进行3x3的逐层卷积（深度卷积DW），在每个通道上进行单独的卷积
                #----------------
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),
                #----------------  
                # 进行1x1的逐点卷积PW，将通道数降低到out_c
                #----------------
                nn.Conv2d(hidden_dim, out_c, 1, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.conv = nn.Sequential(
                #----------------  
                # 进行1x1的逐点卷积PW，将通道数升高到hidden_dim
                #----------------
                nn.Conv2d(in_c, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),
                #----------------
                # 进行3x3的逐层卷积（深度卷积DW），在每个通道上进行单独的卷积
                #----------------
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(True),
                #----------------  
                # 进行1x1的逐点卷积PW，将通道数降低到out_c
                #----------------
                nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False)
            )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_class=1000, input_size=224, width_multi=1):
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        in_channel, last_channel = 32, 1280
        inverted_residual_setting = [
            # t, c, n, s
            # t是放大倍数，c是输出通道数，n是block数，s是步长
            [1, 16, 1, 1], # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2], # 256, 256, 16 -> 128, 128, 24   2
            [6, 32, 3, 2], # 128, 128, 24 -> 64, 64, 32     4
            [6, 64, 4, 2], # 64, 64, 32 -> 32, 32, 64       7
            [6, 96, 3, 1], # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2], # 32, 32, 96 -> 16, 16, 160     14
            [6, 320, 1, 1], # 16, 16, 160 -> 16, 16, 320
        ]

        assert input_size % 32 == 0
        in_channel = int(in_channel * width_multi)
        self.last_channel = int(last_channel * width_multi) if width_multi > 1 else last_channel
        # 512, 512, 3 -> 256, 256, 32
        self.features = [conv_bn(3, in_channel, 2)]

        for t, c, n, s in inverted_residual_setting:
            out_channel = int(c * width_multi)
            for i in range(n):
                # 每一层的第一个block的步长为s，后面的block步长都是1
                if i == 0:
                    self.features.append(block(in_channel, out_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(in_channel, out_channel, 1, expand_ratio=t))
                in_channel = out_channel
        
        self.features.append(conv_1x1_bn(in_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # 最后一层是分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_class)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias.data is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         n = m.weight.size(1)
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.bias, 0, 0.01)
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                # nn.init.normal_(m.bias)

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(num_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'), strict=False)
    return model

if __name__ == '__main__':
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)
    print(len(model.features))
        



