# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""conv weight standarlization."""
import torch.nn as nn
import torch.nn.functional as F


def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    """Conv2d with weight standarlization.

    :param input: input feature map
    :type input: torch.Tensor
    :param weight: weight of conv layer
    :type weight: torch.Tensor
    :param bias: bias
    :type bias: torch.Tensor
    :param stride: conv stride
    :type stride: int
    :param padding: num of padding
    :type padding: int
    :param dilation: num of dilation
    :type dilation: int
    :param groups: num of group
    :type groups: int
    :param eps: weight eps
    :type eps: float
    :return: feature map after weight standarlization
    :rtype: torch.Tensor
    """
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):
    """Conv2d with weight standarlization."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        """Init conv2d with weight standarlization.

        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: num of padding
        :param dilation: num of dilation
        :param groups: num of groups
        :param bias: bias
        :param eps: eps
        """
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        """Forward function of conv2d with weight standarlization."""
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)
