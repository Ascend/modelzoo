#! -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from __future__ import division, print_function

import numpy as np
import argparse
import cv2
import random
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import json
import os,time
from abc import ABCMeta
from abc import abstractproperty
from abc import abstractmethod
import math
import itertools as it
import coco_metric
import collections

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default = 'coco_official_2017/tfrecord/val2017*',
                        help = """the data path""")
    parser.add_argument('--val_json_file', default='coco_official_2017/annotations/instances_val2017.json',
                        help="""the val json file path""")
    parser.add_argument('--result_path', default='results/ssd_resnet34',
                        help="""the result file path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def top_k(input, k=1, sorted=True):
    """Top k max pooling
    Args:
        input(ndarray): convolutional feature in heigh x width x channel format
        k(int): if k==1, it is equal to normal max pooling
        sorted(bool): whether to return the array sorted by channel value
    Returns:
        ndarray: k x (height x width)
        ndarray: k
    """
    ind = np.argpartition(input, -k)[..., -k:]
    def get_entries(input, ind, sorted):
        if len(ind.shape) == 1:
            if sorted:
                ind = ind[np.argsort(-input[ind])]
            return input[ind], ind
        output, ind = zip(*[get_entries(inp, id, sorted) for inp, id in zip(input, ind)])
        return np.array(output), np.array(ind)
    return get_entries(input, ind, sorted)

def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
    scores_trans = np.transpose(scores_in, (0, 2, 1))
    top_k_scores, top_k_indices = top_k(scores_trans, k = pre_nms_num_detections)
    return np.transpose(top_k_scores, (0, 2, 1)), np.transpose(top_k_indices, (0, 2, 1))


def readResult(image_list, result_path, labels_list):
    dataOutput = []
    count = 0
    for image_name in image_list:
            file = image_name.split(".")[0]
            classes = np.fromfile(('{}/davinci_{}_output0.bin').format(result_path, file), dtype='float32').reshape(1, 8732, 81)
            boxes = np.fromfile(('{}/davinci_{}_output1.bin').format(result_path, file), dtype='float32').reshape(1, 8732, 4)
            #boxes = np.fromfile(('{}/davinci_{}_output1.bin').format(result_path, file), dtype='float32').reshape(2,8732)
            pred_scores, indices = select_top_k_scores(classes,200)

            dataOutput.append({"pred_box": boxes[0],
                            "source_id": labels_list[count]['source_id'],
                            "indices": indices[0],
                            "pred_scores": pred_scores[0],
                            "raw_shape": labels_list[count]['raw_shape']})
            count = count + 1
    return dataOutput

def _load_images_info(images_info_file):
  """Loads object annotation JSON file."""
  f = open(images_info_file, encoding='utf-8')
  info_dict = json.load(f)

  img_to_obj_annotation = collections.defaultdict(list)
  for annotation in info_dict['annotations']:
    image_id = annotation['image_id']
    img_to_obj_annotation[image_id].append(annotation)
  return info_dict['images'],img_to_obj_annotation


def get_image_obj(images_info_file, input_images):
    f = open(images_info_file, encoding='utf-8')
    info_dict = json.load(f)
    img_obj = collections.defaultdict(list)
    img_info_list = []
    image_list_new = []
    for image in info_dict['images']:
        img_info = {}
        image_name = image['file_name']
        if image_name not in input_images:
            continue
        img_info['source_id'] = image['id']
        img_info['raw_shape'] = [image['height'], image['width'], 3]
        img_info_list.append(img_info)
        image_list_new.append(image_name)

    return img_info_list, image_list_new

def _read_inputImage(filename):
    image_list = []
    if os.path.isdir(filename):
        for file in os.listdir(filename):
            file = file.split('.')[0] + '.jpg'
            print(file)
            image_list.append(file)
    return image_list


def main():
    args = parse_args()

    image_list = _read_inputImage(args.data_path)
    image_obj, image_list = get_image_obj(args.val_json_file, image_list)

    dataOutput = readResult(image_list, args.result_path, image_obj)

    coco_gt = coco_metric.create_coco(
        args.val_json_file, use_cpp_extension=False)
    coco_metric.compute_map(
        dataOutput,
        coco_gt,
        use_cpp_extension=False,
        nms_on_tpu=False)

if __name__ == '__main__':
    main()
