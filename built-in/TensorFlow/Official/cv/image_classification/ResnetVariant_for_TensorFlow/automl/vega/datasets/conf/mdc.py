# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from .base import BaseConfig


class Mdc2dDetCommonConfig(BaseConfig):
    """Default Dataset config for Mdc2dDet."""

    train_ann_file = None
    valid_ann_file = None
    gt_coco_file = None
    img_prefix = None
    num_classes = 7
    batch_size = 1
    num_workers = 1
    shuffle = True
    distributed = True
    img_scale = dict(train=[(910, 512), (2560, 1440)], test=[(1824, 1024)], valid=[(1824, 1024)])
    multiscale_mode = 'range'  # using multiscale
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    size_divisor = 32
    flip_ratio = 0.5
    with_mask = False
    with_crowd = False
    with_label = True
    proposal_file = False
    num_max_proposals = 1000
    with_semantic_seg = False
    seg_prefix = False
    seg_scale_factor = 1
    extra_aug = False
    resize_keep_ratio = True
    skip_img_without_anno = True
    test_mode = False


class Mdc2dDetTrainConfig(Mdc2dDetCommonConfig):
    """Default Dataset config for Mdc2dDet."""

    pass


class Mdc2dDetValConfig(Mdc2dDetCommonConfig):
    """Default Dataset config for Mdc2dDet."""

    pass


class Mdc2dDetTestConfig(Mdc2dDetCommonConfig):
    """Default Dataset config for Mdc2dDet."""

    pass


class Mdc2dDetConfig(object):
    """Default Dataset config for Mdc2dDet."""

    common = Mdc2dDetCommonConfig
    train = Mdc2dDetTrainConfig
    val = Mdc2dDetValConfig
    test = Mdc2dDetTestConfig
