# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""All negative assigner use max iou."""
from abc import ABCMeta
import torch
from ..bbox_overlap import bbox_overlaps
from .assign_result import AssignResult
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.net_utils import NetTypes


@NetworkFactory.register(NetTypes.UTIL)
class MaxIoUAllNegAssigner(metaclass=ABCMeta):
    """All negative assigner use max iou."""

    def __init__(self, desc):
        """Init Max iou all neg assigner.

        :param desc: config dict
        """
        super().__init__()
        self.pos_iou_thr = desc['pos_iou_thr']
        self.neg_iou_thr = desc['neg_iou_thr']
        self.min_pos_iou = desc['min_pos_iou'] if 'min_pos_iou' in desc else .0
        self.gt_max_assign_all = desc['gt_max_assign_all'] if 'gt_max_assign_all' in desc else True
        self.ignore_iof_thr = desc['ignore_iof_thr'] if 'ignore_iof_thr' in desc else -1
        self.ignore_wrt_candidates = desc['ignore_wrt_candidates'] if 'ignore_wrt_candidates' in desc else True

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign.

        :param bboxes: bboxes
        :param gt_bboxes: ground truth boxes
        :param gt_bboxes_ignore: ground truth boxes need to be ignored
        :param gt_labels: ground truth labels
        :return: assign result
        """
        bboxes = bboxes[:, :4]
        if gt_bboxes.size(0) == 0:
            num_bboxes = bboxes.size(0)
            assigned_gt_inds = bboxes.new_full(
                (num_bboxes,), 0, dtype=torch.long)
            if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                    gt_bboxes_ignore.numel() > 0):
                if self.ignore_wrt_candidates:
                    ignore_overlaps = bbox_overlaps(
                        bboxes, gt_bboxes_ignore, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
                else:
                    ignore_overlaps = bbox_overlaps(
                        gt_bboxes_ignore, bboxes, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
                assigned_gt_inds[ignore_max_overlaps > self.ignore_iof_thr] = -1
            return AssignResult(0, assigned_gt_inds, gt_labels)
        else:
            overlaps = bbox_overlaps(gt_bboxes, bboxes)
            if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                    gt_bboxes_ignore.numel() > 0):
                if self.ignore_wrt_candidates:
                    ignore_overlaps = bbox_overlaps(
                        bboxes, gt_bboxes_ignore, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
                else:
                    ignore_overlaps = bbox_overlaps(
                        gt_bboxes_ignore, bboxes, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
                overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
            assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
            return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign wrt overlaps.

        :param overlaps: overlaps
        :param gt_labels: ground truth labels
        :return: assign result
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds = overlaps.new_full(
            (num_bboxes,), -1, dtype=torch.long)
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
