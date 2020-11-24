# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined roi align use torchvision."""
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align as tv_roi_align


class RoIAlign(nn.Module):
    """Roi align."""

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 use_torchvision=False):
        """Init roi align.

        :param out_size: output size
        :param spatial_scale: spatial scale
        :param sample_num: sample num
        :param use_torchvision: if use torchvision
        """
        super(RoIAlign, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        """Forward compute.

        :param features: input feature
        :param rois: input roi
        :return: result
        """
        result = tv_roi_align(features, rois, _pair(self.out_size),
                              self.spatial_scale, self.sample_num)
        return result.detach()
