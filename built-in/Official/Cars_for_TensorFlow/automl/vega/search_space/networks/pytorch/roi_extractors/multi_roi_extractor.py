# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Multi RoI Extractor."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from .single_level import SingleRoIExtractor


@NetworkFactory.register(NetTypes.ROI_EXTRACTOR)
class MultiRoIExtractor(Network):
    """Multi RoI Extractor."""

    def __init__(self, desc):
        """Init Multi RoI Extractor.

        :param desc: config dict
        """
        super(MultiRoIExtractor, self).__init__()
        self.header_num = desc['header_num']
        self.roi_layer_list = desc['roi_layer_list']
        self.out_channels_list = desc['out_channels_list']
        self.featmap_strides_list = desc['featmap_strides_list']
        self.finest_scale = desc['finest_scale'] if 'finest_scale' in desc else 56
        if len(self.roi_layer_list) == 1:
            self.roi_layer_list = [self.roi_layer_list[0] for _ in range(self.header_num)]
        else:
            self.roi_layer_list = self.roi_layer_list
        if len(self.out_channels_list) == 1:
            self.out_channels_list = [self.out_channels_list[0] for _ in range(self.header_num)]
        else:
            self.out_channels_list = self.out_channels_list
        if len(self.featmap_strides_list) == 1:
            self.featmap_strides_list = [self.featmap_strides_list[0] for _ in range(self.header_num)]
        else:
            self.featmap_strides_list = self.featmap_strides_list
        roi_layers = [SingleRoIExtractor({'roi_layer': self.roi_layer_list[i],
                                          'out_channels': self.out_channels_list[i],
                                          'featmap_strides': self.featmap_strides_list[i]}) for i in
                      range(self.header_num)]
        self.roi_layers = nn.ModuleList(roi_layers)

    def init_weights(self):
        """Init weights."""
        pass

    def forward(self, feats, rois, task_labels):
        """Forward compute.

        :param feats: input feature map
        :param rois: roi
        :param task_labels: task labels
        :return: box feature
        """
        bbox_feats_all_task = dict()
        for header_idx in range(self.header_num):
            if task_labels[header_idx] > 0:
                cur_bbox_roi_extractor = self.roi_layers[header_idx]
                bbox_feats_all_task[header_idx] = cur_bbox_roi_extractor(
                    feats[:cur_bbox_roi_extractor.num_inputs], rois[header_idx])
        return bbox_feats_all_task
