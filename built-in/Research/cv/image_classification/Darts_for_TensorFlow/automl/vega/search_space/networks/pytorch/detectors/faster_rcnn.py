# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined faster rcnn detector."""
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from vega.search_space.networks.network_desc import NetworkDesc
from ..utils.bbox_utils.transforms import bbox2result, bbox2roi


@NetworkFactory.register(NetTypes.DETECTOR)
class FasterRCNN(Network):
    """Faster RCNN."""

    def __init__(self, desc):
        """Init faster rcnn.

        :param desc: config dict
        """
        super(FasterRCNN, self).__init__()
        self.data_to_cuda = False
        self.num_klass = int(desc["num_klass"])
        self.backbone_cfg = desc["backbone"] if "backbone" in desc else None
        self.neck_cfg = desc["neck"] if "neck" in desc else None
        self.rpn_head_cfg = desc["rpn_head"] if "rpn_head" in desc else None
        self.bbox_roi_extractor_cfg = desc["bbox_roi_extractor"] if "bbox_roi_extractor" in desc else None
        self.bbox_head_cfg = desc["bbox_head"] if "bbox_head" in desc else None
        self.pretrained_cfg = desc["pretrained"] if "pretrained" in desc else None
        self.shared_head_cfg = desc["shared_head"] if "shared_head" in desc else None
        self.train_cfg = desc['train_cfg']
        self.test_cfg = desc['test_cfg']
        self.train_cfg = desc['train_cfg']
        self.test_cfg = desc['test_cfg']
        if self.backbone_cfg is not None:
            self.backbone = NetworkDesc(self.backbone_cfg).to_model()
        if self.neck_cfg is not None:
            self.neck = NetworkDesc(self.neck_cfg).to_model()
            self.with_neck = True
        if self.rpn_head_cfg is not None:
            self.rpn_head = NetworkDesc(self.rpn_head_cfg).to_model()
            self.with_rpn = True
        if self.bbox_head_cfg is not None:
            self.bbox_head = NetworkDesc(self.bbox_head_cfg).to_model()
            self.with_bbox = True
        if self.bbox_roi_extractor_cfg is not None:
            self.bbox_roi_extractor = NetworkDesc(self.bbox_roi_extractor_cfg).to_model()
        if self.shared_head_cfg is not None:
            self.shared_head = NetworkDesc(self.shared_head_cfg).to_model()
            self.with_shared_head = True
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        self.backbone.init_weights()
        if self.with_neck:
            self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Public compute of input.

        :param img: input image
        :return: feature map after backbone and neck compute
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def cuda(self, device=None):
        """Move all model parameters and buffers to the GPU."""
        self.data_to_cuda = True
        self.rpn_head.cuda(device)
        return self._apply(lambda t: t.cuda(device))

    def forward_train(self, input, **kwargs):
        """Forward compute between train process.

        :param input: input data
        :return: losses
        """
        img = input['img']
        img_meta = input['img_meta']
        gt_bboxes = input['gt_bboxes']
        gt_labels = input['gt_labels']
        gt_bboxes_ignore = input['gt_bboxes_ignore']
        gt_labels_ignore = input['gt_labels_ignore']
        img_meta = [img_meta]
        if self.data_to_cuda:
            img = img.cuda()
            gt_bboxes = gt_bboxes.cuda()
            gt_labels = gt_labels.cuda()
            gt_bboxes_ignore = gt_bboxes_ignore.cuda()
            gt_labels_ignore = gt_labels_ignore.cuda()
        x = self.extract_feat(img)
        losses = dict()
        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
        rpn_losses, _, _, _ = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        if self.with_bbox or self.with_mask:
            bbox_assigner = NetworkFactory.get_network(NetTypes.UTIL, self.train_cfg.rcnn.assigner.name)
            bbox_assigner = bbox_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = NetworkFactory.get_network(NetTypes.UTIL, self.train_cfg.rcnn.sampler.name)
            bbox_sampler = bbox_sampler(self.train_cfg.rcnn.sampler)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = bbox_sampler.sample(assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i],
                                                      feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)
        return losses

    def forward_valid(self, input, **kwargs):
        """Forward compute between inference.

        :param input: input data
        :return: box result and labels result
        """
        img = input['img']
        img_meta = input['img_meta']
        rescale = input['rescale'] if 'rescale' in input else False
        if self.data_to_cuda:
            img = img[0].cuda()
        else:
            img = img[0]
        x = self.extract_feat(img)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            proposal_cfg = self.test_cfg.rpn
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            return x
        if self.with_bbox:
            rois = bbox2roi(proposal_list)
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(rois, cls_score, bbox_pred, img_shape, scale_factor,
                                                                   rescale=rescale, cfg=self.test_cfg.rcnn)
            bbox_results = bbox2result(det_bboxes, det_labels, self.num_klass)
            return bbox_results
