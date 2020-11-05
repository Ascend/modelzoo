# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""random sampler."""
from abc import ABCMeta
import torch
from .sample_result import SampleResult
import numpy as np
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.net_utils import NetTypes


@NetworkFactory.register(NetTypes.UTIL)
class RandomSampler(metaclass=ABCMeta):
    """Random sampler."""

    def __init__(self, desc):
        """Init sampler.

        :param desc: config dict
        """
        super().__init__()
        self.num = desc['num']
        self.pos_fraction = desc['pos_fraction']
        self.neg_pos_ub = desc['neg_pos_ub'] if 'neg_pos_ub' in desc else -1
        self.add_gt_as_proposals = desc['add_gt_as_proposals'] if 'add_gt_as_proposals' in desc else True

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples.

        :param assign_result: assign result
        :param num_expected: num expect
        :return: positive object index
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples.

        :param assign_result: assign result
        :param num_expected: negative result
        :return: negative object index
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        :param gallery: gallery
        :param num: num
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        :param assign_result: Bbox assigning results
        :param bboxes: boxes to be sampled from
        :param gt_bboxes: ground truth boxes
        :param gt_labels: ground truth labels
        :return: sample result
        """
        bboxes = bboxes[:, :4]
        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()
        return SampleResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
