# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Head of CurveLaneNas."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.HEAD)
class AutoLaneHead(Network):
    """CurveLaneHead."""

    def __init__(self, desc):
        """Construct head."""
        super(AutoLaneHead, self).__init__()
        base_channel = desc["base_channel"]
        num_classes = desc["num_classes"]
        lane_up_pts_num = desc["up_points"]
        lane_down_pts_num = desc["down_points"]
        self.lane_up_pts_num = lane_up_pts_num
        self.lane_down_pts_num = lane_down_pts_num
        self.num_classes = num_classes

        BatchNorm = nn.BatchNorm2d

        self.conv_up_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
            BatchNorm(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, lane_up_pts_num, kernel_size=1, stride=1)
        )

        self.conv_down_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
            BatchNorm(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, lane_down_pts_num, kernel_size=1, stride=1)
        )

        self.conv_cls_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
            BatchNorm(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, num_classes, kernel_size=1, stride=1)
        )
        for index, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_train(self, input, **kwargs):
        """Forward method of this head."""
        predict_up = self.conv_up_conv(input).permute((0, 2, 3, 1))
        predict_down = self.conv_down_conv(input).permute((0, 2, 3, 1))
        predict_cls = self.conv_cls_conv(input).permute((0, 2, 3, 1)).contiguous()

        predict_loc = torch.cat([predict_down, predict_up], -1).contiguous()

        predict_loc = predict_loc.view(predict_loc.shape[0], -1, self.lane_up_pts_num + self.lane_down_pts_num)
        predict_cls = predict_cls.view(predict_cls.shape[0], -1, self.num_classes)

        result = dict(
            predict_cls=predict_cls,
            predict_loc=predict_loc
        )
        return result

    @property
    def input_shape(self):
        """Output of backbone."""
        return 18, 32
