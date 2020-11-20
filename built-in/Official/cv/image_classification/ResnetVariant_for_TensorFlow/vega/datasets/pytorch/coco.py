# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for coco dataset."""
import os.path as osp
import numpy as np
import cv2
import mmcv
from pycocotools.coco import COCO
from vega.datasets.pytorch.common.dataset import Dataset
from ..transforms import (ImageTransform, BboxTransform, MaskTransform,
                          SegMapTransform, Numpy2Tensor)
from vega.datasets.pytorch.common.utils import to_tensor, random_scale
from vega.core.metrics.pytorch.detection_metric import coco_eval
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.datasets.conf.coco import CocoConfig


@ClassFactory.register(ClassType.DATASET)
class CocoDataset(Dataset):
    """This is the class of coco dataset, which is a subclass of Dateset.

    :param train: `train`, `val` or test
    :type train: str
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = CocoConfig()

    def __init__(self, **kwargs):
        """Construct the CocoDataset class."""
        super(CocoDataset, self).__init__(**kwargs)
        self.dataset_init()
        # self.CLASSES = self.cfg.train.num_classes

    def dataset_init(self):
        """Construct method."""
        # args = self.cfg.dataset
        self.CLASSES = self.args.num_classes
        if self.mode == "train":
            self.ann_file = self.args.train_ann_file
            self.img_prefix = self.args.train_img_prefix
        elif self.mode == "val":
            self.ann_file = self.args.val_ann_file
            self.img_prefix = self.args.val_img_prefix
        elif self.mode == "test":
            self.ann_file = self.args.test_ann_file
            self.img_prefix = self.args.test_img_prefix
        else:
            raise

        self.img_infos = self.load_annotations()
        if self.args.proposal_file is not None:
            self.proposals = self.load_proposals()
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not self.args.test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        # self.img_scales = args.img_scale if isinstance(args.img_scale,
        #           list) else [args.img_scale]

        self.img_scales = self.args.img_scale.train if isinstance(self.args.img_scale.train, list) else [
            self.args.img_scale.train]
        self.val_img_scales = self.args.img_scale.test if isinstance(self.args.img_scale.test, list) else [
            self.args.img_scale.test]

        # assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = self.args.img_norm_cfg

        # multi-scale mode (only applicable for multi-scale training)
        self.multiscale_mode = self.args.multiscale_mode

        # max proposals per image
        self.num_max_proposals = self.args.num_max_proposals
        # flip ratio
        self.flip_ratio = self.args.flip_ratio

        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = self.args.size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = self.args.with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = self.args.with_crowd
        # with label is False for RPN
        self.with_label = self.args.with_label
        # with semantic segmentation (stuff) annotation or not
        self.with_seg = self.args.with_semantic_seg
        # prefix of semantic segmentation map path
        self.seg_prefix = self.args.seg_prefix
        # rescale factor for segmentation maps
        self.seg_scale_factor = self.args.seg_scale_factor
        # in test mode or not
        self.test_mode = self.args.test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.seg_transform = SegMapTransform(self.size_divisor)
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        # if args.extra_aug is not None:
        #     self.extra_aug = ExtraAugmentation(**extra_aug)
        # else:
        self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = self.args.resize_keep_ratio

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.img_infos)

    def load_proposals(self):
        """Load proposals."""
        return mmcv.load(self.proposal_file)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Random pool."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def load_annotations(self):
        """Load annotations."""
        self.coco = COCO(self.ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            info['img_id'] = i
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        """Get ann information."""
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        self.ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info()

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self):
        """Parse bbox and mask annotation."""
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if self.with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(self.ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
                gt_labels_ignore.append(0)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if self.with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            gt_labels_ignore = np.array(gt_labels_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
            gt_labels_ignore = np.zeros((0), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore, labels_ignore=gt_labels_ignore)

        if self.with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Prepare an image for training.

        :param idx: index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        img_info = self.img_infos[idx]
        # load image
        # img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        img_path = osp.join(self.img_prefix, img_info['filename'])
        # dataset, img_path, anno_path = img_info.rsplit('  ')
        # if img_path.startswith('s3'):
        #     img = cv2.imdecode(np.fromstring(mox.file.read(img_path, binary=True), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.imread(img_path)
        if img is None:
            print('load image error:', img_info)
            return None
        #
        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None
        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)
        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        # data = dict(
        #     img=DC(to_tensor(img), stack=True),
        #     img_meta=DC(img_meta, cpu_only=True),
        #     gt_bboxes=DC(to_tensor(gt_bboxes)))

        data = dict(
            img=to_tensor(img),
            img_meta=img_meta,
            gt_bboxes=to_tensor(gt_bboxes))

        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
            # data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)
            data['gt_semantic_seg'] = to_tensor(gt_seg)
        if self.proposals is not None:
            proposals, scores = self.process_proposals(idx)
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
            # data['proposals'] = DC(to_tensor(proposals))
            data['proposals'] = to_tensor(proposals)
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']
            gt_labels_ignore = ann['labels_ignore']
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
            # data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
            data['gt_bboxes_ignore'] = to_tensor(gt_bboxes_ignore)
            data['gt_labels_ignore'] = to_tensor(gt_labels_ignore)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)
            # data['gt_masks'] = DC(gt_masks, cpu_only=True)
            data['gt_masks'] = gt_masks
        if self.with_label:
            # data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_labels'] = to_tensor(gt_labels)
        return (data, to_tensor(0))

    def process_proposals(self, idx):
        """Process proposals.

        :param idx: index
        :type idx: int
        :return: proposals, score
        :rtype: list
        """
        proposals = self.proposals[idx][:self.num_max_proposals]
        # TODO: Handle empty proposals properly. Currently images with
        # no proposals are just ignored, but they can be used for
        # training in concept.
        if len(proposals) == 0:
            return None
        if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        if proposals.shape[1] == 5:
            scores = proposals[:, 4, None]
            proposals = proposals[:, :4]
        else:
            scores = None
        return proposals, scores

    def prepare_test_img(self, idx):
        """Prepare an image for testing.

        :param idx: index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        img_info = self.img_infos[idx]
        # img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        img_path = osp.join(self.img_prefix, img_info['filename'])
        # dataset, img_path, anno_path = img_info.rsplit('  ')
        # if img_path.startswith('s3'):
        #     img = cv2.imdecode(np.fromstring(mox.file.read(img_path, binary=True), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.imread(img_path)
        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        # for scale in self.img_scales:
        for scale in self.val_img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            # img_metas.append(DC(_img_meta, cpu_only=True))
            img_metas.append(_img_meta)
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                # img_metas.append(DC(_img_meta, cpu_only=True))
                img_metas.append(_img_meta)
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return (data, img_info['img_id'])

    def evaluate(self, outputs, eval_types, out_file):
        """Evaluate method for the coco dataset.

        :param outputs: the predict result by the model
        :type outputs: json file
        :param eval_types: choice:['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints']
        :type eval_types: str
        :param out_file: out_file
        :type out_file: json file
        :return: evallute result
        :rtype: float
        """

        def xyxy2xywh(bbox):
            """Transform the bbox coordinate to [x,y ,w,h].

            :param bbox: the predict bounding box coordinate
            :type bbox: list
            :return: [x,y,w,h]
            :rtype: list
            """
            _bbox = bbox.tolist()
            return [
                _bbox[0],
                _bbox[1],
                _bbox[2] - _bbox[0],
                _bbox[3] - _bbox[1],
            ]

        result_files = dict()
        json_results = []
        for idx in range(len(self.img_infos)):
            img_id = self.img_ids[idx]
            result = outputs[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
            outputs['bbox'] = f'{out_file}.bbox.json'
            outputs['proposal'] = f'{out_file}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        eval_results = coco_eval(result_files, eval_types, self.coco)
        return eval_results
