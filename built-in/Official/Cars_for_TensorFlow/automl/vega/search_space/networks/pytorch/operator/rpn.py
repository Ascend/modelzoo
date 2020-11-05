# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch.nn.functional as F
import torch.nn as nn
import torch
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.pytorch.utils.anchor_utils.anchor_target import AnchorTarget
from vega.search_space.networks.pytorch.utils.bbox_utils.anchor_generator import AnchorGenerator
from vega.core.common.config import Config
from functools import partial
import numpy as np
from six.moves import map, zip
from vega.search_space.networks.pytorch.losses.reduce_loss import weighted_loss


@NetworkFactory.register(NetTypes.Operator)
class RpnClsLossInput(nn.Module):
    """Rpn input."""

    def __init__(self):
        super(RpnClsLossInput, self).__init__()

    def forward(self, x):
        """Get cls score and bbox preds."""
        cls_scores = x[0]
        bbox_preds = x[1]
        return cls_scores, bbox_preds


@NetworkFactory.register(NetTypes.Operator)
class RpnLossInput(nn.Module):
    """Rpn loss input."""

    def __init__(self):
        super(RpnLossInput, self).__init__()

    def forward(self, x):
        """Get cls score."""
        cls_scores = x[2][0]
        bbox_preds = x[2][1]
        gt_bboxes = x[0]['gt_bboxes'].cuda()
        img_metas = [x[0]['img_meta']]
        gt_bboxes_ignore = x[0]['gt_bboxes_ignore'].cuda()
        return cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore


@NetworkFactory.register(NetTypes.Operator)
class AnchorTargetOp(nn.Module):
    """Anchor Target."""

    def __init__(self, target_means=None, target_stds=None, num_classes=2, use_sigmoid_cls=False, cfg=None,
                 sampling=True):
        self.target_means = target_means or (.0, .0, .0, .0)
        self.target_stds = target_stds or (1.0, 1.0, 1.0, 1.0)
        self.label_channels = num_classes if use_sigmoid_cls else 1
        self.cfg = Config({'assigner': {'name': 'MaxIoUAllNegAssigner', 'pos_iou_thr': 0.7,
                                        'neg_iou_thr': tuple([-1, 0.3]), 'min_pos_iou': 0.3, 'ignore_iof_thr': 0.5},
                           'sampler': {'name': 'RandomSampler', 'num': 256, 'pos_fraction': 0.5, 'neg_pos_ub': -1,
                                       'add_gt_as_proposals': False}, 'allowed_border': 0, 'pos_weight': -1,
                           'debug': False})
        self.sampling = sampling
        super(AnchorTargetOp, self).__init__()

    def forward(self, x):
        """Create X=(anchor_list,valid_flag_list,gt_bboxes,img_metas,)."""
        anchor_list, valid_flag_list, original_anchors, gt_bboxes, img_metas, gt_bboxes_ignore = x
        #  out=(labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,num_total_neg).
        return AnchorTarget(anchor_list, valid_flag_list, gt_bboxes, img_metas, self.target_means,
                            self.target_stds,
                            self.cfg, gt_bboxes_ignore_list=gt_bboxes_ignore,
                            gt_labels_list=None,
                            label_channels=self.label_channels,
                            sampling=self.sampling)


@NetworkFactory.register(NetTypes.Operator)
class Anchors(nn.Module):
    """Get anchors according to feature map sizes."""

    def __init__(self, anchor_base_sizes_cfg=None, anchor_scales=None, anchor_ratios=None, anchor_strides=None):
        self.anchor_base_sizes_cfg = anchor_base_sizes_cfg
        self.anchor_scales = anchor_scales or [8, 16, 32]
        self.anchor_ratios = anchor_ratios or [0.5, 1.0, 2.0]
        self.anchor_strides = anchor_strides or [4, 8, 16, 32, 64]
        self.anchor_base_sizes = list(
            self.anchor_strides) if self.anchor_base_sizes_cfg is None else self.anchor_base_sizes_cfg
        super(Anchors, self).__init__()

    def forward(self, x):
        """Create anchor."""
        cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore = x
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            anchor_generators.append(AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list, multi_level_anchors, gt_bboxes, img_metas, gt_bboxes_ignore


def multi_apply(func, *args, **kwargs):
    """Multi apply.

    :param func: function
    :param args: args of function
    :return: result
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@NetworkFactory.register(NetTypes.Operator)
class RpnClsLoss(nn.Module):
    """Rpn Class Loss."""

    def __init__(self, out_channels=2):
        super(RpnClsLoss, self).__init__()
        self.loss_cls = CustomCrossEntropyLoss()
        self.loss_bbox = CustomSmoothL1Loss()
        self.out_channels = out_channels

    def forward(self, x):
        """Get x."""
        (cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_pos, num_total_neg,
         num_total_samples) = x
        losses_cls, losses_bbox = multi_apply(self.loss, cls_score, bbox_pred, labels, label_weights, bbox_targets,
                                              bbox_weights, num_total_samples=num_total_samples)
        return losses_cls, losses_bbox

    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets, bbox_weights, num_total_samples):
        """Get loss."""
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox


@NetworkFactory.register(NetTypes.Operator)
class CustomCrossEntropyLoss(nn.Module):
    """Cross Entropy Loss."""

    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean',
                 loss_weight=1.0):
        """Init Cross Entropy loss.

        :param desc: config dict
        """
        super(CustomCrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        if self.use_sigmoid:
            self.loss_function = binary_cross_entropy
        elif self.use_mask:
            self.loss_function = mask_cross_entropy
        else:
            self.loss_function = cross_entropy

    def forward(self, cls_score, label, weight, avg_factor, reduction_override=None, **kwargs):
        """Forward compute."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.loss_function(cls_score, label, weight, reduction=reduction,
                                                         avg_factor=avg_factor, **kwargs)
        return loss_cls


@NetworkFactory.register(NetTypes.Operator)
class CustomSmoothL1Loss(nn.Module):
    """Smooth L1 Loss."""

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        """Init smooth l1 loss."""
        super(CustomSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward compute.

        :param pred: predict
        :param target: target
        :param weight: weight
        :param avg_factor: avg factor
        :param reduction_override: reduce override
        :return: loss
        """
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.numel() > 0:
            loss_bbox = self.loss_weight * smooth_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)
            return loss_bbox
        else:
            return torch.FloatTensor([0.0]).cuda()


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth l1 loss.

    :param pred: predict
    :param target: target
    :param beta: beta
    :return: loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    """Cross entropy losses.

    :param pred: predict result
    :param label: gt label
    :param weight: weight
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    loss = F.cross_entropy(pred, label, reduction='none')
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    """Expand binary labels.

    :param labels: labels
    :param label_weights: label weights
    :param label_channels: label channels
    :return: binary label and label weights
    """
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    """Binary cross entropy loss.

    :param pred: predict result
    :param label: gt label
    :param weight:  weight
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    """Mask cross entropy loss.

    :param pred: predict result
    :param target: target
    :param label: gt label
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(pred_slice, target, reduction='mean')[None]


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Weight reduce loss.

    :param loss: losses
    :param weight: weight
    :param reduction: reduce function
    :param avg_factor: avg factor
    :return: loss
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss compute.

    :param loss: losses
    :param reduction: reduce funtion
    :return: loss
    """
    reduction_function = F._Reduction.get_enum(reduction)
    if reduction_function == 0:
        return loss
    elif reduction_function == 1:
        return loss.mean()
    elif reduction_function == 2:
        return loss.sum()
