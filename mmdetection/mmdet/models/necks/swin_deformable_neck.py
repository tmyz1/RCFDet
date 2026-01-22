# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from torch import Tensor
from mmdet.models.necks import YOLOXPAFPN


@MODELS.register_module()
class swin_deformable_neck(BaseModule):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 kernel_size: int = 3,
                 reduction: int = 16):
        super(swin_deformable_neck, self).__init__()
        self.out_channels = out_channels
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        self.RGB_CBAM = nn.ModuleList()
        self.IR_CBAM = nn.ModuleList()
        #空间通道注意力
        for in_channel in in_channels:
            RGB_CBAM_block = CBAM_BLOCK(planes=in_channel)
            IR_CBAM_block = CBAM_BLOCK(planes=in_channel)
            self.RGB_CBAM.append(RGB_CBAM_block)
            self.IR_CBAM.append(IR_CBAM_block)
        #特征金字塔
        self.rgb_fpn = YOLOXPAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_csp_blocks=len(in_channels),
        )
        self.ir_fpn = YOLOXPAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_csp_blocks=len(in_channels),
        )
        self.align_blocks = nn.ModuleList([
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU')
            )
            for _ in range(len(in_channels))
        ])

    def forward(self, inputs):
        rgb, ir, a = inputs
        a = a.view(-1, 1, 1, 1)
        #空间通道注意力
        for i in range(len(rgb)):
            rgb[i] = self.RGB_CBAM[i](rgb[i])
            ir[i] = self.IR_CBAM[i](ir[i])
        #特征金字塔
        rgb = self.rgb_fpn(rgb)
        ir = self.ir_fpn(ir)

        # 相加融合
        fused = []
        for i in range(len(rgb)):
            f = rgb[i] * a + ir[i] * (1 - a)
            f = self.align_blocks[i](f)
            fused.append(f)

        return fused

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM_BLOCK(nn.Module):
    def __init__(self, planes):
        super(CBAM_BLOCK, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def view(image:Tensor,name:str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 300, 300)
    temp = image[0, 15].detach().cpu().numpy()
    temp = np.uint8((temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-8) * 255)
    cv2.imshow(name, temp)




