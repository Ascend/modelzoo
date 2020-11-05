# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Single RoI Extractor."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from ..utils import RoIAlign


@NetworkFactory.register(NetTypes.ROI_EXTRACTOR)
class SingleRoIExtractor(Network):
    """Single RoI Extractor."""

    def __init__(self, desc):
        """Init Single RoI Extractor.

        :param desc: config dict
        """
        super(SingleRoIExtractor, self).__init__()
        self.layer_cfg = desc['roi_layer']
        self.out_channels = desc['out_channels']
        self.featmap_strides = desc['featmap_strides']
        cfg = self.layer_cfg.copy()
        cfg.pop('type')
        self.roi_layers = nn.ModuleList([RoIAlign(spatial_scale=1 / s, **cfg) for s in self.featmap_strides])
        self.finest_scale = desc['finest_scale'] if 'finest_scale' in desc else 56
        self.num_inputs = len(self.featmap_strides)

    def init_weights(self):
        """Init weights."""
        pass

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        :param rois: input roi
        :param num_levels: Total level number.
        :return: Level index (0-based) of each RoI
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        """Forward compute.

        :param feats: input feature map
        :param rois: input rois
        :return: roi feature
        """
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)
        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(rois.size()[0], self.out_channels, out_size, out_size)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
        return roi_feats
