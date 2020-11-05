# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined share conv fc layer."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from ..blocks.conv_module import ConvModule


@NetworkFactory.register(NetTypes.SHARED_HEAD)
class ShareConvFc(Network):
    """Share conv fc layer."""

    def __init__(self, desc):
        """Init share conv fc layer.

        :param desc: config dict
        """
        super(ShareConvFc, self).__init__()
        self.num_shared_convs = desc['num_shared_convs'] if 'num_shared_convs' in desc else 0
        self.num_shared_fcs = desc['num_shared_fcs'] if 'num_shared_fcs' in desc else 0
        self.with_avg_pool = desc['with_avg_pool'] if 'with_avg_pool' in desc else False
        self.roi_feat_size = desc['roi_feat_size'] if 'roi_feat_size' in desc else 7
        self.in_channels = desc['in_channels'] if 'in_channels' in desc else 256
        self.conv_out_channels = desc['conv_out_channels'] if 'conv_out_channels' in desc else 256
        self.fc_out_channels = desc['fc_out_channels'] if 'fc_out_channels' in desc else 1024
        self.conv_cfg = desc['conv_cfg'] if 'conv_cfg' in desc else {'type': 'Conv'}
        self.norm_cfg = desc['norm_cfg'] if 'norm_cfg' in desc else {"type": "BN", "requires_grad": True}
        self.norm_eval = desc['norm_eval'] if 'norm_eval' in desc else True
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels)
        self.shared_out_channels = last_layer_dim
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            ):
        """Add conv and fc.

        :param num_branch_convs: num of convolution layer
        :param num_branch_fcs: num of fc layers
        :param in_channels: input channel
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self, pretrained=None):
        """Init weights."""
        for module_list in [self.shared_convs, self.shared_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward compute.

        :param x: input feature map
        :return: out put feature map
        """
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x

    def train(self, mode=True):
        """Train config.

        :param mode: if train
        """
        super(ShareConvFc, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
