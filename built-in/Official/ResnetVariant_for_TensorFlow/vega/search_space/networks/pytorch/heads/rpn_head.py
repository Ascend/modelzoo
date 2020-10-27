# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Rpn Head."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.network_desc import NetworkDesc
from ..utils.bbox_utils.anchor_generator import AnchorGenerator
from ..utils.bbox_utils.transforms import delta2bbox
import numpy as np
from ..utils.bbox_utils.map_util import multi_apply
from ..utils.anchor_utils.anchor_target import AnchorTarget
import torch.nn.functional as F


@NetworkFactory.register(NetTypes.HEAD)
class RPNHead(Network):
    """RPN Head."""

    def __init__(self, desc):
        """Init RPN Head."""
        super(RPNHead, self).__init__()
        self.device = None
        self.data_to_cuda = False
        self.num_classes = 2
        self.in_channels = desc['in_channels']
        self.feat_channels = desc['feat_channels'] if 'feat_channels' in desc else 256
        self.anchor_scales = desc['anchor_scales'] if 'anchor_scales' in desc else [8, 16, 32]
        self.anchor_ratios = desc['anchor_ratios'] if 'anchor_ratios' in desc else [0.5, 1.0, 2.0]
        self.anchor_strides = desc['anchor_strides'] if 'anchor_strides' in desc else [4, 8, 16, 32, 64]
        self.anchor_base_sizes_cfg = desc['anchor_base_sizes'] if 'anchor_base_sizes' in desc else None
        self.target_means = desc['target_means'] if 'target_means' in desc else (.0, .0, .0, .0)
        self.target_stds = desc['target_stds'] if 'target_stds' in desc else (1.0, 1.0, 1.0, 1.0)
        self.loss_cls_cfg = desc['loss_cls'] if 'loss_cls' in desc else {"name": 'CrossEntropyLoss',
                                                                         "use_sigmoid": True, "loss_weight": 1.0}
        self.loss_bbox_cfg = desc['loss_bbox'] if 'loss_bbox' in desc else {"name": 'SmoothL1Loss', "beta": 1.0 / 9.0,
                                                                            "loss_weight": 1.0}
        self.loss_cls_cfg = {"modules": ['loss'], 'loss': self.loss_cls_cfg}
        self.loss_bbox_cfg = {"modules": ['loss'], 'loss': self.loss_bbox_cfg}
        self.loss_cls = NetworkDesc(self.loss_cls_cfg).to_model()
        self.loss_bbox = NetworkDesc(self.loss_bbox_cfg).to_model()
        self.anchor_base_sizes = list(
            self.anchor_strides) if self.anchor_base_sizes_cfg is None else self.anchor_base_sizes_cfg
        self.use_sigmoid_cls = self.loss_cls_cfg.get('use_sigmoid', False)
        self.sampling = self.loss_cls_cfg['loss']['name'] not in ['FocalLoss', 'GHMC']
        self.cls_out_channels = self.num_classes
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))
        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        """Init rpn layers."""
        if self.feat_channels > 0:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
            self.rpn_cls = nn.Conv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels, 1)
            self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        else:
            self.rpn_cls = nn.Conv2d(self.in_channels,
                                     self.num_anchors * self.cls_out_channels, 1)
            self.rpn_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        """Init weights."""
        if self.feat_channels > 0:
            nn.init.normal_(self.rpn_conv.weight, std=0.01)
            if hasattr(self.rpn_conv, 'bias') and self.rpn_conv.bias is not None:
                nn.init.constant_(self.rpn_conv.bias, 0)
        nn.init.normal_(self.rpn_cls.weight, std=0.01)
        if hasattr(self.rpn_cls, 'bias') and self.rpn_cls.bias is not None:
            nn.init.constant_(self.rpn_cls.bias, 0)
        nn.init.normal_(self.rpn_reg.weight, std=0.01)
        if hasattr(self.rpn_reg, 'bias') and self.rpn_reg.bias is not None:
            nn.init.constant_(self.rpn_reg.bias, 0)

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg, rescale=False):
        """Get boxes result.

        :param cls_scores: class score
        :param bbox_preds: box predict
        :param img_metas: image meta info
        :param cfg: config
        :param rescale: if need rescale
        :return: boxes result
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                               self.anchor_strides[i],
                                                               self.device)
                        for i in range(num_levels)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, cfg, rescale=False):
        """Get boxes result single.

        :param cls_scores: class scores
        :param bbox_preds: box predict
        :param mlvl_anchors: multi level anchor
        :param img_shape: image shaore
        :param scale_factor: scale factor
        :param cfg: config
        :param rescale: if need resclae
        :return: box proposal
        """
        from mmdet.ops import nms
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero(
                    (w >= cfg.min_bbox_size) & (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        :param featmap_sizes: Multi-level feature map sizes
        :type featmap_sizes: list of tuple
        :param img_metas: image meta info
        :type img_metas: list of dict
        :return: anchor list, valid flag list, multi level anchor
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], self.device)
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
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), self.device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list, multi_level_anchors

    def forward_single(self, x):
        """Forward compute single.

        :param x: input feature map
        :return: rpn class score, rpn box predict result
        """
        if self.feat_channels > 0:
            x = self.rpn_conv(x)
            x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def cuda(self, device=None):
        """Move all model parameters and buffers to the GPU."""
        self.data_to_cuda = True
        self.device = 'cuda'
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feats):
        """Forward compute.

        :param feats: input feature map
        :return: rpn_cls_score, rpn_bbox_pred
        """
        return multi_apply(self.forward_single, feats)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        """Loss compute single.

        :param cls_score: class score
        :param bbox_pred: box predict result
        :param labels: labels
        :param label_weights: label weights
        :param bbox_targets: box targets
        :param bbox_weights: box weights
        :param num_total_samples: total num of sample result
        :param cfg: config
        :return: loss of class and box
        """
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, cfg, gt_bboxes_ignore=None):
        """Loss compute.

        :param cls_scores: class score
        :param bbox_preds: box predict result
        :param gt_bboxes: ground truth box
        :param img_metas: image meta info
        :param cfg: config
        :param gt_bboxes_ignore:ground truch box need to be ignored
        :return: dict of loss
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list, original_anchors = self.get_anchors(featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = AnchorTarget(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels,
            sampling=self.sampling,
            device=self.device)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox), original_anchors, self.num_anchors, cls_scores
