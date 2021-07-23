# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
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


from __future__ import division, print_function

import numpy as np
import argparse
import random
import json
import os
from PIL import Image
import PIL

parser = argparse.ArgumentParser(description="SSD_mobilenet test single image test procedure.")

parser.add_argument("--save_json_path", type=str, default="./result.json",
                    help="The path of the result.json.")

parser.add_argument("--result_file_path", type=str, default="./result",
                    help="The path of inference result bin file.")

parser.add_argument("--save_json", type=bool, default=True,
                    help="whether to save detected-result cocolike json")
parser.add_argument("--img_conf_path", type=str, default="./data/img_info",
                    help="The path of img config path, include the image name ,width and height.")
args = parser.parse_args()

def get_default_dict():
    return {"image_id": -1, "category_id": -1, "bbox": [], "score": 0}

def get_valid_data(out_boxes,out_scores,out_classes,out_num_detections):
    out_boxes = out_boxes.reshape(-1,4)
    out_scores = out_scores.reshape(-1,100)
    out_classes = out_classes.reshape(-1,100)
    picked_boxes, picked_score, picked_label = [], [], []
    filter_boxes, filter_scores, filter_label = [],[],[]
    for i in range(int(out_num_detections[0])):
        filter_boxes.append(out_boxes[i])
        filter_scores.append(out_scores[:,i])
        filter_label.append(out_classes[:,i])
        
    if len(filter_boxes) == 0:
        return None, None, None
    picked_boxes.append(filter_boxes)
    picked_score.append(filter_scores)
    picked_label.append(filter_label)

    out_boxes = np.concatenate(picked_boxes, axis=0)
    out_scores = np.concatenate(picked_score, axis=0)
    out_classes = np.concatenate(picked_label, axis=0)

    return out_boxes, out_scores, out_classes


def get_ori_config():
    img_conf = args.img_conf_path
    with open(img_conf, 'r')as f:
        img_info_list = f.read().split('\n')[:-1]
    img_info_dict = {}
    for i in img_info_list:
        tmp_list = i.split(' ')
        idx = int(tmp_list[0])
        imgName = tmp_list[1]
        ori_width = int(tmp_list[2])
        ori_height = int(tmp_list[3])
        img_info_dict[idx] = {
            'imgName' : imgName,
            'im_width' : ori_width,
            'im_height': ori_height,
        }
    return img_info_dict

def data_process(img_info_dict,json_out):
    test_len = len(img_info_dict.keys())
    for test_idx in range(test_len):
        img_name = img_info_dict[test_idx]['imgName']
        img_temp = str(img_name)
        out_boxes = np.fromfile(('{}/davinci_{}_output2.bin').format(args.result_file_path, img_temp), dtype='float16').reshape(1,100,4)
        out_scores = np.fromfile(('{}/davinci_{}_output1.bin').format(args.result_file_path, img_temp), dtype='float16').reshape(1,100)
        out_classes = np.fromfile(('{}/davinci_{}_output3.bin').format(args.result_file_path, img_temp), dtype='float16').reshape(1,100)
        out_num_detections = np.fromfile(('{}/davinci_{}_output0.bin').format(args.result_file_path, img_temp),dtype='float16').reshape(1)
        out_boxes , out_scores, out_classes = get_valid_data(out_boxes,out_scores,out_classes,out_num_detections)
        im_width = img_info_dict[test_idx]['im_width']
        im_height = img_info_dict[test_idx]['im_height']

        img_temp = img_temp.split('.')
        if args.save_json:
            if out_boxes is None :
                print(img_temp)
            else:
                for i in range(len(out_boxes)):
                    ymin,xmin,ymax,xmax = out_boxes[i]
                    left = xmin * im_width
                    right = xmax * im_width
                    top = ymin * im_height
                    bottom =  ymax * im_height
                    x = left
                    y = top
                    width = right - left
                    height = bottom - top
                    s = out_scores[i]
                    c = out_classes[i]
                    t_dict = get_default_dict()
                    t_dict['image_id'] = int(img_temp[0][img_temp[0].rindex('_') + 1: ])
                    t_dict['category_id'] = int(c)
                    t_dict['bbox'] = [int(i) for i in [x, y, width, height]]
                    t_dict['score'] = float(s)
                    json_out.append(t_dict)

def post_process():
    img_info_dict = get_ori_config()
    json_out = []
    data_process(img_info_dict,json_out)
    if args.save_json:
        with open(args.save_json_path, 'w')as f:
            json.dump(json_out, f)

if __name__ == '__main__':
    post_process()
