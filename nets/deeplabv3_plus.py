from turtle import forward
from unittest import result
import torch
from torch import nn 
import os
import math
import torch.nn.functional as F 
from mobilenetv2 import mobilenetv2
from Xception import xception


class MobileNetV2(nn.Module):
    def __init__(self, downsample=8, pretrained=False):
        super(MobileNetV2, self).__init__()
        from functools import partial
        self.model = mobilenetv2(pretrained)
        model           = mobilenetv2(pretrained)
        #-----------------------------------#
        #   获取backbone中提取主干特征的层
        #-----------------------------------#
        self.features   = model.features[:-1]
        self.total_idx = len(self.features)
        #---------------------------#
        #   提取浅层特征的层的索引
        #---------------------------#
        self.down_idx = [2, 4, 7, 14]

        if downsample == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilation=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilation=4))
        elif downsample == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilation=2))


    def _nostride_dilate(self, m, dilation):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation // 2, dilation // 2)
                    m.padding = (dilation // 2, dilation // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return x, low_level_features


#-----------------------------------------#
#   ASPP特征提取模块,空洞空间金字塔池化
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, in_c, out_c, atrous, bn_mom=0.01):
        super(ASPP, self).__init__()
        #   
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=bn_mom),
            nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, padding=atrous[0], dilation=atrous[0], bias=False),
            nn.BatchNorm2d(out_c, bn_mom),
            nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, padding=atrous[1], dilation=atrous[1], bias=False),
            nn.BatchNorm2d(out_c, momentum=bn_mom),
            nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, padding=atrous[2], dilation=atrous[2], bias=False),
            nn.BatchNorm2d(out_c, momentum=bn_mom),
            nn.ReLU(True),
        )
        self.branch5_conv = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(out_c, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_c * 5, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c, momentum=bn_mom),
            nn.ReLU(True),
        )

    def forward(self, x):
        b, c, r, c = x.size()

        #-------------------------------#
        #   前四个分支的空洞卷积
        #-------------------------------#
        features1 = self.branch1(x)
        features2 = self.branch2(x)
        features3 = self.branch3(x)
        features4 = self.branch4(x)

        #------------------------------------#
        #   第五个分支进行全局平均池化+卷积   
        #------------------------------------#
        features5 = torch.mean(x, 2, True)
        features5 = torch.mean(features5, 3, True)
        features5 = self.branch5_conv(features5)
        features5 = self.branch5_bn(features5)
        features5 = self.branch5_relu(features5)
        features5 = F.interpolate(features5,(r, c), None, 'bilinear', True)

        #----------------------------------------#
        #   将5个分支的特征拼接起来，然后做卷积
        #----------------------------------------#
        features = torch.cat((features1, features2, features3, features4, features5), dim=1)
        result = self.conv_cat(features)

        return result

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_class, backbone='mobilenet', pretrained=False, downsample=16):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'xception':
            #---------------------------------#
            #   获得两个特征层
            #   浅层特征    [128, 128, 256]
            #   主干特征    [30, 30, 2048]
            #---------------------------------#
            self.backbone = xception(pretrained=pretrained, downsample=downsample)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == 'mobilenet':
            #---------------------------------#
            #   获得两个特征层
            #   浅层特征    [128, 128, 24]
            #   主干特征    [30, 30, 320]
            #---------------------------------#
            self.backbone = MobileNetV2(downsample=downsample, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use Xception or mobilenet'.format(backbone))
        
        #---------------------------------#
        #   ASPP模块
        #   利用不同的空洞率进行特征提取
        #---------------------------------#
        if downsample == 16:
            atrous = [6, 12, 18]
        elif downsample == 8:
            atrous = [12, 24, 36]
        else:
            raise ValueError('Unsupported downsample - `{}`, Use 8 or 16'.format(downsample))
        self.aspp = ASPP(in_channels, 256, atrous)

        #----------------#
        #   浅层特征边
        #----------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.conv_cat = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

        self.cls = nn.Conv2d(256, num_class, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #---------------------------------------------#
        #   获取浅层特征low_level_feature和主体特征x
        #---------------------------------------------#
        x, low_level_features = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        #------------------------------------------#
        #   对主干特征进行上采样
        #   将浅层特征和主干特征连接在一起进行卷积
        #------------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.conv_cat(torch.cat((x, low_level_features), dim=1))
        x = self.cls(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x

# x = torch.randn((1, 3, 299, 299))
# model1 = DeepLabV3Plus(1000)
# y, low = model1.backbone(x)
# print("y:", y.shape)
# print("low level features:", low.shape)
# model2 = MobileNetV2(downsample=16,pretrained=False)
# y, low = model2(x)
# print("y:", y.shape)
# print("low level features:", low.shape)
# model_x = DeepLabV3Plus(1000, 'xception')
# y, low = model_x.backbone(x)
# print("y:", y.shape)
# print("low level features:", low.shape)




