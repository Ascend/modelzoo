# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Bound box Head."""
import torch
import torch.nn as nn
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from ..utils.bbox_utils.transforms import delta2bbox, bbox2delta
from ..utils.bbox_utils.map_util import multi_apply
from vega.search_space.networks.network_desc import NetworkDesc
import torch.nn.functional as F


@NetworkFactory.register(NetTypes.HEAD)
class BBoxHead(Network):
    """Bound Box Head."""

    def __init__(self, desc):
        """Init box head.

        :param desc: config dict
        """
        super(BBoxHead, self).__init__()
        self.with_avg_pool = desc['with_avg_pool'] if 'with_avg_pool' in desc else False
        self.with_cls = desc['with_cls'] if 'with_cls' in desc else True
        self.with_reg = desc['with_reg'] if 'with_reg' in desc else True
        self.roi_feat_size = desc['roi_feat_size'] if 'roi_feat_size' in desc else 7
        self.in_channels = desc['in_channels'] if 'in_channels' in desc else 256
        self.num_classes = desc['num_classes'] if 'num_classes' in desc else 10
        self.target_means = desc['target_means'] if 'target_means' in desc else [0., 0., 0., 0.]
        self.target_stds = desc['target_stds'] if 'target_stds' in desc else [0.1, 0.1, 0.2, 0.2]
        self.reg_class_agnostic = desc['reg_class_agnostic'] if 'reg_class_agnostic' in desc else False
        self.roi_feat_area = self.roi_feat_size * self.roi_feat_size
        self.loss_cls_cfg = desc['loss_cls'] if 'loss_cls' in desc else {"name": 'CrossEntropyLoss',
                                                                         "use_sigmoid": False, "loss_weight": 1.0}
        self.loss_bbox_cfg = desc['loss_bbox'] if 'loss_bbox' in desc else {"name": 'SmoothL1Loss', "beta": 1.0,
                                                                            "loss_weight": 1.0}
        self.loss_cls_cfg = {"modules": ['loss'], 'loss': self.loss_cls_cfg}
        self.loss_bbox_cfg = {"modules": ['loss'], 'loss': self.loss_bbox_cfg}
        self.loss_cls = NetworkDesc(self.loss_cls_cfg).to_model()
        self.loss_bbox = NetworkDesc(self.loss_bbox_cfg).to_model()
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        """Init weight."""
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x, **kwargs):
        """Forward compute.

        :param x: input feature map
        :return: class score and bbox result
        :rtype: torch.Tensor
        """
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        """Get class Regression targets.

        :param sampling_results: input proposal
        :type sampling_results: torch.Tensor
        :param gt_bboxes: ground truth boxes
        :type gt_bboxes: torch.Tensor
        :param gt_labels: ground truch labels
        :type gt_labels: torch.Tensor
        :param rcnn_train_cfg: train config
        :type rcnn_train_cfg: dict
        :return: class Regression targets
        :rtype: torch.Tensor
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = self.bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def bbox_target(self, pos_bboxes_list,
                    neg_bboxes_list,
                    pos_gt_bboxes_list,
                    pos_gt_labels_list,
                    cfg,
                    reg_classes=1,
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0],
                    concat=True):
        """Compute box target.

        :param pos_bboxes_list: positive box
        :type pos_bboxes_list: torch.Tensor
        :param neg_bboxes_list: negative box
        :type neg_bboxes_list: torch.Tensor
        :param pos_gt_bboxes_list: positive ground truth bbox
        :type pos_gt_bboxes_list: torch.Tensor
        :param pos_gt_labels_list: positive ground truth labels
        :type pos_gt_labels_list: torch.Tensor
        :param cfg: rcnn config
        :type cfg: dict
        :param reg_classes: num of classes
        :type reg_classes: int
        :param target_means: target means
        :type target_means: list
        :param target_stds: target stds
        :type target_stds: list
        :param concat: if concat result
        :type concat: bool
        :return: labels, label weights, bbox targets, bbox weights
        :rtype: troch.Tensor
        """
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self.bbox_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=cfg,
            reg_classes=reg_classes,
            target_means=target_means,
            target_stds=target_stds)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def bbox_target_single(self, pos_bboxes,
                           neg_bboxes,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           cfg,
                           reg_classes=1,
                           target_means=[.0, .0, .0, .0],
                           target_stds=[1.0, 1.0, 1.0, 1.0]):
        """Compute box target single.

        :param pos_bboxes: positive box
        :type pos_bboxes: torch.Tensor
        :param neg_bboxes: negative box
        :type neg_bboxes: torch.Tensor
        :param pos_gt_bboxes: positive ground truth bbox
        :type pos_gt_bboxes: torch.Tensor
        :param pos_gt_labels: positive ground truth labels
        :type pos_gt_labels: torch.Tensor
        :param cfg: rcnn config
        :type cfg: dict
        :param reg_classes: num of classes
        :type reg_classes: int
        :param target_means: target means
        :type target_means: list
        :param target_stds: target stds
        :type target_stds: list
        :return: labels, label weights, bbox targets, bbox weights
        :rtype: troch.Tensor
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                          target_stds)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        """Get detection boxes result.

        :param rois: roi feature
        :type rois: torch.Tensor
        :param cls_score: class score
        :type cls_score: torch.Tensor
        :param bbox_pred: boxes predict result
        :type bbox_pred: torch.Tensor
        :param img_shape: image shape
        :type img_shape: list
        :param scale_factor: scale factor
        :type scale_factor: int
        :param rescale: if need rescale
        :type rescale: bool
        :param cfg: predict config
        :type cfg: dict
        :return: boxes, labels
        :rtype: torch.Tensor
        """
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            from mmdet.core import multiclass_nms
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        :param rois: sampled RoIs per image.
        :type rois: torch.Tensor
        :param labels: labels
        :type labels: torch.Tensor
        :param bbox_preds: box predict
        :type bbox_preds: torch.Tensor
        :param pos_is_gts: Flags indicating if each positive bbox is a gt bbox.
        :type pos_is_gts: list of torch.Tensor
        :param img_metas: image info
        :type img_metas: list of dict
        :return: Refined bboxes of each image in a mini-batch.
        :rtype: list of torch.Tensor
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)
        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]
            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            bboxes_list.append(bboxes[keep_inds])
        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        :param rois: roi boxes
        :type rois: torch.Tensor
        :param label: label
        :type label: torch.Tensor
        :param bbox_pred: bbox predict
        :type bbox_pred: torch.Tensor
        :param img_meta: image meta info
        :type img_meta: dict
        :return: Regressed bboxes, the same shape as input rois.
        :rtype: torch.Tensor
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """Loss compute.

        :param cls_score: class score
        :param bbox_pred: boxes predict
        :param labels: labels
        :param label_weights: label weights
        :param bbox_targets: box targets
        :param bbox_weights: box weights
        :param reduction_override: reduction override
        :return: losses
        """
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = self.accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    def accuracy(self, pred, target, topk=1):
        """Accuracy compute.

        :param pred: predict result
        :param target: targets
        :param topk: topk
        :return: accuracy
        """
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
            return_single = True
        else:
            return_single = False
        maxk = max(topk)
        _, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / pred.size(0)))
        return res[0] if return_single else res
