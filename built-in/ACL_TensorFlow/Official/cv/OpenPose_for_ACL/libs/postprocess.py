#!/usr/bin/python3
# coding=utf-8
# Copied from
# https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess/
# In-depth article about the algorithm:
# https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
# Refactored to reduce complexity
# changed to process peak coordinates from tensorflow::ops::Where
# instead of peaks map from python tf.Where
# ======================================================================
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
import json
import os
import sys
import time
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils import Logging, TfPoseEstimator
from utils import model_wh, write_coco_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tensorflow Openpose Inference Postprocess for NPU'
    )
    parser.add_argument(
        '--resize',
        type=str,
        default='0x0',
        help='if provided, resize images before they are post-processed.'
             'Recommends : 432x368 or 656x368 or 1312x736'
    )
    parser.add_argument(
        '--resize-out-ratio',
        type=float,
        default=8.0,
        help='if provided, resize heatmaps before they are post-processed.'
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
        '--data-idx',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./input/'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/'
    )
    args = parser.parse_args()

    logger = Logging()

    coco_year_list = ['2014', '2017']
    if args.coco_year not in coco_year_list:
        logger.error('COCO year should be one of %s' % str(coco_year_list))
        sys.exit(-1)

    image_dir = os.path.join(args.coco_dir, 'val%s' % args.coco_year)
    coco_json_file = os.path.join(
        args.coco_dir,
        'annotations/person_keypoints_val%s.json' % args.coco_year
    )

    coco_gt = COCO(coco_json_file)
    cat_ids = coco_gt.getCatIds(catNms=['person'])
    keys = coco_gt.getImgIds(catIds=cat_ids)

    if args.data_idx > 0:
        keys = keys[:args.data_idx]
    logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))
    write_json = './%s_%s_%0.1f.json' % (args.model, args.resize, args.resize_out_ratio)

    width, height = model_wh(args.resize)
    if width == 0 or height == 0:
        estimator = TfPoseEstimator(target_size=(432, 368))
    else:
        estimator = TfPoseEstimator(target_size=(width, height))

    result = []
    tqdm_keys = tqdm(keys)
    for i, k in enumerate(tqdm_keys):
        img_meta = coco_gt.loadImgs(k)[0]
        img_idx = img_meta['id']
        t = time.time()
        file_name = os.path.join(args.output_dir, "%s_output_00_000.bin" % os.path.splitext(img_meta['file_name'])[0])
        humans = estimator.inference(file_name=file_name,
                                     resize_to_default=(width > 0 and height > 0),
                                     upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        all_score = 0
        for human in humans:
            keypoints = write_coco_json(human, img_meta['width'], img_meta['height'])
            item = {
                'image_id': img_idx,
                'category_id': 1,
                'keypoints': keypoints,
                'score': human.score
            }
            result.append(item)
            all_score += item['score']

        avg_score = all_score / len(humans) if len(humans) > 0 else 0
        tqdm_keys.set_postfix(OrderedDict({'inference time': elapsed, 'score': avg_score}))

    fp = open(write_json, 'w')
    json.dump(result, fp)
    fp.close()

    coco_dt = coco_gt.loadRes(write_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.imgIds = keys
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    logger.info(''.join(['%2.4f |' % x for x in coco_eval.stats]))
