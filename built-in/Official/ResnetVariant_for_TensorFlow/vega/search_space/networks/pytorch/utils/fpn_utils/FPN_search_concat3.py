# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""FPN module for neck."""
import torch
import torch.nn as nn
from .conv_module import ConvModule


class FPN_Search(nn.Module):
    """FPN generate Module."""

    def __init__(self,
                 out_channels=128,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 in_channels=None,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 fpn_arch_str=None):
        """Init the class.

        :param out_channels: output channels for every fpn layer
        :type: int
        :param num_outs: outputs count.
        :type: int
        :param start_level: start index of input channels
        :type: int
        :param end_level: end index of input channels
        """
        super(FPN_Search, self).__init__()
        # assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            # assert end_level <= len(in_channels)
            # assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.fpn_arch_str = fpn_arch_str
        self.concat_convs = nn.ModuleList()
        self.c34_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.c24_maxpool = nn.MaxPool2d(kernel_size=5, stride=4, padding=1)
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels * 2,
                out_channels * 2,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.concat_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Init the weight by default way."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.xavier_init(m, distribution='uniform')

    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        """Xavier init the weights."""
        # assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def decoder_fpn_arch(self):
        """Decode fpn arch."""
        fpn_arch = []
        block_arch = []
        for i in self.fpn_arch_str:
            if i == '-':
                fpn_arch.append(block_arch)
                block_arch = []
            else:
                block_arch.append(int(i))
        fpn_arch.append(block_arch)
        return fpn_arch

    def forward(self, inputs):
        """Forward method."""
        # assert len(inputs) == len(self.in_channels)
        build_out = []
        # target_w, target_h = 32, 18
        fpn_arch = self.decoder_fpn_arch()
        for i in range(len(fpn_arch)):
            # input1, input2, output = fpn_arch[i][0], fpn_arch[i][1], fpn_arch[i][2]
            # input1, input2, _ = fpn_arch[i][0], fpn_arch[i][1], fpn_arch[i][2]
            input1, input2 = fpn_arch[i][0], fpn_arch[i][1]
            laterals = []
            laterals.append(self.lateral_convs[input1](inputs[input1]))  # input 1
            laterals.append(self.lateral_convs[input2](inputs[input2]))  # input 2

            # sum of the two input
            if input1 == 0:
                laterals[0] = self.c24_maxpool(laterals[0])
            elif input1 == 1:
                laterals[0] = self.c34_maxpool(laterals[0])
            if input2 == 0:
                laterals[1] = self.c24_maxpool(laterals[1])
            elif input2 == 1:
                laterals[1] = self.c34_maxpool(laterals[1])

            build_out.append(self.fpn_convs[i](torch.cat((laterals[0], laterals[1]), 1)))

        outs = torch.cat((inputs[2], torch.cat((build_out[0], build_out[1]), 1)), 1)
        return outs


def fpn_search(fpn_arch, in_channels):
    """Fpn search warpper."""
    return FPN_Search(in_channels=in_channels, out_channels=64, num_outs=4, fpn_arch_str=fpn_arch)
