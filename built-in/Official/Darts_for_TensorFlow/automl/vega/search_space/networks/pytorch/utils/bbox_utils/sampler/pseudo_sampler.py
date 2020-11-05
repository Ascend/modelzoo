# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Pseudo sampler."""
import torch
from .sample_result import SampleResult
from abc import ABCMeta
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.net_utils import NetTypes


@NetworkFactory.register(NetTypes.UTIL)
class PseudoSampler(metaclass=ABCMeta):
    """Pseudo sampler."""

    def __init__(self):
        """Init sampler."""
        super().__init__()

    def _sample_pos(self, **kwargs):
        """Sample positive object."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative object."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Sample.

        :param assign_result: assign result
        :param bboxes: boxes
        :param gt_bboxes: ground boxes
        :return: sample result
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SampleResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                       assign_result, gt_flags)
        return sampling_result
