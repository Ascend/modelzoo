#!/usr/bin/python3
# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from utils import Logging
from utils import model_wh, read_img_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tensorflow Openpose Inference Preprocess for NPU'
    )
    parser.add_argument(
        '--resize',
        type=str,
        default='0x0',
        help='if provided, resize images before they are post-processed.'
             'Recommends : 432x368 or 656x368 or 1312x736'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cmu',
        help='support model: cmu or mobilenet_thin or mobilenet_v2_large'
    )
    parser.add_argument(
        '--coco-year',
        type=str,
        default='2014'
    )
    parser.add_argument(
        '--coco-dir',
        type=str,
        default='./dataset/coco/'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./input/'
    )
    args = parser.parse_args()

    logger = Logging()

    coco_year_list = ['2014', '2017']
    if args.coco_year not in coco_year_list:
        logger.error('coco_year should be one of %s' % str(coco_year_list))
        sys.exit(-1)

    image_dir = os.path.join(args.coco_dir, 'val%s' % args.coco_year)
    coco_json_file = os.path.join(
        args.coco_dir,
        'annotations/person_keypoints_val%s.json' % args.coco_year
    )
    coco_gt = COCO(coco_json_file)
    cat_ids = coco_gt.getCatIds(catNms=['person'])
    keys = coco_gt.getImgIds(catIds=cat_ids)

    logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))

    width, height = model_wh(args.resize)
    if width == 0 or height == 0:
        width = 432
        height = 368
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tqdm_keys = tqdm(keys)
    for i, k in enumerate(tqdm_keys):
        img_meta = coco_gt.loadImgs(k)[0]
        img_idx = img_meta['id']
        img_name = os.path.join(image_dir, img_meta['file_name'])
        image = read_img_file(img_name, width, height)
        if image is None:
            logger.error('image not found, path=%s' % img_name)
            sys.exit(-1)
        # ori shape (h, w, c), transpose to (1, h, w, c)
        image = image.reshape(1, height, width, 3).astype(np.float32)
        bin_name = img_meta['file_name'].split('.')[0] + '.bin'
        image.tofile(os.path.join(output_dir, bin_name))
