from audioop import bias
from locale import atof
from pyexpat import model
from turtle import forward
from unittest import skip
from soupsieve import select
import torch
from torch import nn 
import os
import math
import torch.utils.model_zoo as model_zoo
from yaml import load


bn_mom = 0.0003

class SeparableConv(nn.Module):
    def __init__(self, 
                 in_c, 
                 out_c,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dialtion=1,
                 bias=False,
                 activate_first=True):
        super(SeparableConv, self).__init__()
        self.sepconv = nn.Sequential(*[
            nn.ReLU(True),
            nn.Conv2d(in_c, in_c, kernel_size, stride, padding, dialtion, groups=in_c, bias=bias),
            nn.BatchNorm2d(in_c, momentum=bn_mom),
            nn.ReLU(True),
            nn.Conv2d(in_c, out_c, 1, 1, 0, 1, groups=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=bn_mom),
            nn.ReLU(True)
        ])
        if activate_first:
            self.sepconv = self.sepconv[:-1]
        else:
            self.sepconv = self.sepconv[1:]

        # self.relu0 = nn.ReLU(True)
        # self.depthwise = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dialtion, groups=in_c, bias=bias)
        # self.bn1 = nn.BatchNorm2d(out_c, momentum=bn_mom)
        # self.relu1 = nn.ReLU(True)
        # self.pointwise = nn.Conv2d(out_c, out_c, 1, 1, 0, 1, groups=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_c, momentum=bn_mom)
        # self.relu2 = nn.ReLU(True)
        # self.activate_first = activate_first

    def forward(self, x):
        # if self.activate_first:
        #     x = self.relu0(x)
        # x = self.depthwise(x)
        # x = self.bn1(x)
        # if not self.activate_first:
        #     x = self.relu1(x)
        # x = self.pointwise(x)
        # x = self.bn2(x)
        # if not self.activate_first:
        #     x = self.relu2(x)

        x = self.sepconv(x)
        return x

class Block(nn.Module):
    def __init__(self, in_c, out_c, reps, stride=1, dilation=1, grow_first=True, activate_first=True, inplace=True):
        super(Block, self).__init__()
        # if atrous == None:
        #     atrous = [1] * 3
        # elif isinstance(atrous, int):
        #     atrous_list = [atrous] * 3
        #     atrous = atrous_list
        self.head_relu = True

        if in_c != out_c or stride != 1:
            self.skip = nn.Conv2d(in_c, out_c, 1, stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_c, bn_mom)
            # self.head_relu = False
        else:
            self.skip = None
        
        self.hook_layer = None
        # grow_first控制第一个可分离卷积层是否将输出通道提高到out_c
        if grow_first:
            mid_c = out_c
        else:
            mid_c = in_c

        blocks = []
        blocks.append(SeparableConv(in_c, mid_c, 3, stride=1, padding=dilation, dialtion=dilation, bias=False, activate_first=activate_first))

        for i in range(reps - 2):
            blocks.append(SeparableConv(mid_c, mid_c, 3, stride=1, padding=dilation, dialtion=dilation, bias=False, activate_first=activate_first))

        blocks.append(SeparableConv(mid_c, out_c, 3, stride=stride, padding=dilation, dialtion=dilation, bias=False, activate_first=activate_first))

        self.block_feature = nn.Sequential(*blocks)

        # self.sepconv1 = SeparableConv(in_c, mid_c, 3, stride=1, padding=dilation, dialtion=dilation, bias=False, activate_first=activate_first)
        # self.sepconv2 = SeparableConv(mid_c, out_c, 3, stride, padding=dilation, dialtion=dilation, bias=False, activate_first=activate_first)
        # self.sepconv3 = SeparableConv(out_c, out_c, 3, stride, padding=dilation, dialtion=dilation, bias=False, activate_first=activate_first)

    def forward(self, x):
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        
        # x = self.sepconv1(x)
        # x = self.sepconv2(x)
        # self.hook_layer = x
        # x = self.sepconv3(x)
        x = self.block_feature[:-1](x)
        self.hook_layer = x
        x = self.block_feature[-1](x)
        x += skip
        return x 

class Xception(nn.Module):
    
    def __init__(self, downsample_factor):
        super(Xception, self).__init__()
        stride_list = None
        if downsample_factor == 8:
            # 下采样倍数是8时，在Entry Flow中的block里选两个进行下采样
            # 在这里对前两个block都进行了下采样，只需要控制第三个block是否下采样
            # 空洞率并不影响输出特征的尺寸大小，下采样8倍时需要更大的感受野，所以将exit_block的空洞率设置的大一点
            entry_block_3 = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        elif downsample_factor == 16:
            # 下采样倍数是8时，在Entry Flow中的block里选两个进行下采样
            # 在这里对前两个block都进行了下采样，只需要控制第三个block是否下采样
            # 空洞率并不影响输出特征的尺寸大小
            # 空洞率并不影响输出特征的尺寸大小，下采样16倍时需要更大的感受野，所以将exit_block的空洞率设置的小一点
            entry_block_3 = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        else:
            raise ValueError("output_stride=%d is not supported"%os)

        # Entry Flow
        self.block0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_mom)
        )

        # self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        # self.relu1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)

        self.block1 = Block(64, 128, 2, 2)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 768, 2, entry_block_3)

        # Middle Flow
        self.MiddleFlow = nn.Sequential(
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation),
            Block(768, 768, 3, dilation=middle_block_dilation)
        )

        # Exit Flow
        self.ExitFlow = nn.Sequential(
            Block(768, 1024, 2, stride=1, dilation=exit_block_dilations[0]),
            SeparableConv(1024, 1536, 3, 1, padding=exit_block_dilations[1], dialtion=exit_block_dilations[1]),
            SeparableConv(1536, 1536, 3, 1, padding=exit_block_dilations[1], dialtion=exit_block_dilations[1]),
            SeparableConv(1536, 2048, 3, 1, padding=exit_block_dilations[1], dialtion=exit_block_dilations[1])
        )

        self._initialize_weights()

    def forward(self, x):

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        low_level_features = self.block2.hook_layer
        x = self.block3(x)
        # low_level_features = x
        x = self.MiddleFlow(x)
        x = self.ExitFlow(x)

        return x, low_level_features

    # 初始化参数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir)

def xception(pretrained=False, downsample=16):
    model = Xception(downsample)
    if pretrained:
        model.load_state_dict(load_url(' '), strict=False)
    return model


# model = xception(downsample=16)
# # print(model)
# x = torch.randn(1, 3, 299, 299)
# y, low = model(x)
# print(y.shape, low.shape)



