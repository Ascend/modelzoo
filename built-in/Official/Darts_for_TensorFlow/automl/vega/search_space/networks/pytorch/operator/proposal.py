# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch.nn as nn
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.pytorch.utils import MaxIoUAllNegAssigner
from vega.search_space.networks.pytorch.utils import RandomSampler
from vega.search_space.networks.pytorch.utils.proposal import get_bboxes
from vega.search_space.networks.pytorch.utils import AnchorGenerator


@NetworkFactory.register(NetTypes.Operator)
class Proposals(nn.Module):
    """Propsals for Faster-Rcnn."""

    def __init__(self, desc):
        self.desc = desc
        self.anchor_generators = []
        for anchor_base in list(desc.anchor_strides):
            self.anchor_generators.append(AnchorGenerator(anchor_base, desc.anchor_scales, desc.anchor_ratios))
        super(Proposals, self).__init__()

    def forward(self, x):
        """Create forward."""
        cls_scores = x[2][0]
        bbox_preds = x[2][1]
        img_metas = [x[0]['img_meta']]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_anchors = []
        for i in range(len(cls_scores)):
            anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.desc.anchor_strides[i])
            mlvl_anchors.append(anchors)
        return get_bboxes(cls_scores, bbox_preds, img_metas, mlvl_anchors, self.desc), x[0]


@NetworkFactory.register(NetTypes.Operator)
class Assigner(nn.Module):
    """Assigner for Faster-RCNN."""

    def __init__(self, desc):
        self.desc = desc
        self.assigner = MaxIoUAllNegAssigner(desc.get("assigner"))
        self.bbox_sampler = RandomSampler(desc.get("sampler"))
        super(Assigner, self).__init__()

    def forward(self, x):
        """Assign x."""
        bboxes = x[0]
        gt_bboxes = x[1]['gt_bboxes']
        gt_bboxes_ignore = x[1]['gt_bboxes_ignore']
        gt_labels = x[1]['gt_bboxes']['gt_labels']
        results = []
        for i in range(bboxes.size(0)):
            assign_result = self.assigner.assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)
            results.append(self.bbox_sampler.sample(assign_result, bboxes[i], gt_bboxes[i], gt_labels[i],
                                                    feats=[lvl_feat[i][None] for lvl_feat in x]))
        return results
