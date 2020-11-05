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


class CocoCommonConfig(BaseConfig):
    """Default Dataset config for Coco."""

    data_root = None
    train_ann_file = None
    val_ann_file = None
    test_ann_file = None
    train_img_prefix = None
    val_img_prefix = None
    test_img_prefix = None
    num_classes = 81
    num_workers = 1
    distributed = True
    img_scale = dict(train=(720, 480), test=(1280, 720), val=(1280, 720))
    # img_scale = dict(train=(1333, 800), test=(1333, 800), val=(1333, 800))
    multiscale_mode = 'range'  # using multiscale
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    size_divisor = 32
    flip_ratio = 0.5
    with_mask = False
    with_crowd = False
    with_label = True
    proposal_file = None
    num_max_proposals = 1000
    with_semantic_seg = False
    seg_prefix = False
    seg_scale_factor = 1
    extra_aug = False
    resize_keep_ratio = True
    skip_img_without_anno = True
    test_mode = False
    imgs_per_gpu = 1


class CocoTrainConfig(CocoCommonConfig):
    """Default Dataset config for Coco train."""

    with_crowd = True


class CocoValConfig(CocoCommonConfig):
    """Default Dataset config for Coco val."""

    flip_ratio = 0
    with_crowd = True


class CocoTestConfig(CocoCommonConfig):
    """Default Dataset config for Coco val."""

    flip_ratio = 0
    with_crowd = True


class CocoConfig(object):
    """Default Dataset config for Coco."""

    common = CocoCommonConfig
    train = CocoTrainConfig
    val = CocoValConfig
    test = CocoTestConfig
