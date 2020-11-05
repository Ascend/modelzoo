# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch
from ..utils.bbox_utils.transforms import delta2bbox
from vega.core.common.config import Config


def get_bboxes(cls_scores, bbox_preds, img_metas, mlvl_anchors, cfg):
    """Get boxes result.

    :param cls_scores: class score
    :param bbox_preds: box predict
    :param img_metas: image meta info
    :param cfg: config
    :param rescale: if need rescale
    :return: boxes result
    """
    num_levels = len(cls_scores)
    result_list = []
    for img_id in range(len(img_metas)):
        cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
        img_shape = img_metas[img_id]['img_shape']
        proposals = get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, cfg)
        result_list.append(proposals)
    return result_list


def get_bboxes_single(cls_scores, bbox_preds, mlvl_anchors, img_shape, cfg):
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
    cfg = Config(cfg)
    mlvl_proposals = []
    for idx in range(len(cls_scores)):
        rpn_cls_score = cls_scores[idx]
        rpn_bbox_pred = bbox_preds[idx]
        assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
        anchors = mlvl_anchors[idx]
        rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
        if cfg.use_sigmoid_cls:
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
        proposals = delta2bbox(anchors, rpn_bbox_pred, cfg.target_means,
                               cfg.target_stds, img_shape)
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
