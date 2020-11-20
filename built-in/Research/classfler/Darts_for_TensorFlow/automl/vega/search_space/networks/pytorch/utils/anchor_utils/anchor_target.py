# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Anchor target."""
import torch
from ..bbox_utils.map_util import multi_apply
from ..bbox_utils.sampler import PseudoSampler
from ..bbox_utils.transforms import bbox2delta
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.net_utils import NetTypes


def AnchorTarget(anchor_list,
                 valid_flag_list,
                 gt_bboxes_list,
                 img_metas,
                 target_means,
                 target_stds,
                 cfg,
                 gt_bboxes_ignore_list=None,
                 gt_labels_list=None,
                 label_channels=1,
                 sampling=True,
                 unmap_outputs=True,
                 device='cuda'):
    """Compute regression and classification targets for anchors.

    :param anchor_list: Multi level anchors of each image
    :param valid_flag_list: Multi level valid flags of each image
    :param gt_bboxes_list: Ground truth bboxes of each image
    :param img_metas: Meta info of each image
    :param target_means: Mean value of regression targets
    :param target_stds: Std value of regression targets
    :param cfg: RPN train configs
    :return: tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
        anchor_target_single,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs,
        device=device)
    if any([labels is None for labels in all_labels]):
        return None
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    :param target: target
    :param num_level_anchors: num level anchors
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         device='cuda'):
    """Anchor target single.

    :param flat_anchors: float anchor
    :param valid_flags: valid flag
    :param gt_bboxes: gt boxes
    :param gt_bboxes_ignore: gt boxes need to be ignored
    :param gt_labels: gt labels
    :param img_meta: image meta info
    :param target_means: target mean
    :param target_stds: target std
    :param cfg: config
    :param label_channels: label channels
    :param sampling: sample
    :param unmap_outputs: unmap outputs
    :return: tuple
    """
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border,
                                       device)
    if not inside_flags.any():
        return (None,) * 6
    anchors = flat_anchors[inside_flags, :]
    if sampling:
        bbox_assigner = NetworkFactory.get_network(
            NetTypes.UTIL, cfg.assigner.name)
        bbox_assigner = bbox_assigner(cfg.assigner)
        bbox_sampler = NetworkFactory.get_network(
            NetTypes.UTIL, cfg.sampler.name)
        bbox_sampler = bbox_sampler(cfg.sampler)
        assign_result = bbox_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        sampling_result = bbox_sampler.sample(
            assign_result, anchors, gt_bboxes, gt_labels)
    else:
        bbox_assigner = NetworkFactory.get_network(
            NetTypes.UTIL, cfg.assigner.name)
        bbox_assigner = bbox_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes)
    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0, device='cuda'):
    """Anchor inside flags.

    :param flat_anchors: flat anchors
    :param valid_flags: valid flags
    :param img_shape: image meta info
    :param allowed_border: if allow border
    :return: inside flags
    """
    img_h, img_w = img_shape[:2]
    if device == 'cuda':
        img_h = img_h.cuda()
        img_w = img_w.cuda()
    img_h = img_h.float()
    img_w = img_w.float()
    valid_flags = valid_flags.bool()

    if allowed_border >= 0:
        inside_flags = (valid_flags & (flat_anchors[:, 0] >= -allowed_border) & (
            flat_anchors[:, 1] >= -allowed_border) & (
                flat_anchors[:, 2] < img_w + allowed_border) & (
                    flat_anchors[:, 3] < img_h + allowed_border))
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size count).

    :param data: input data
    :param count: count
    :param inds: index
    :param: fill
    :return ret
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
