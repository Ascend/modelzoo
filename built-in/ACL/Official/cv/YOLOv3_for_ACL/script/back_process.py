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

# coding: utf-8

from __future__ import division, print_function

import numpy as np
import argparse
import random
import json
import os

'''
coco weight from official checked 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.555
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.629

'''

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--score_thresh", type=float, default=1e-3,
                    help="score_threshold for test")
parser.add_argument("--nms_thresh", type=float, default=0.55,
                    help="iou_threshold for test")
parser.add_argument("--max_boxes", type=int, default=100,
                    help="max_boxes for test")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--save_json", type=bool, default=True,
                    help="whether to save detected-result cocolike json")
parser.add_argument("--save_json_path", type=str, default="./result.json",
                    help="The path of the result.json.")
#parser.add_argument("--img_info_path", type=str, default="./data/img_info",
                    #help="The path of img_info.")
parser.add_argument("--img_info_path", type=str, default="",
                    help="The path of img_info.")
parser.add_argument("--result_file_path", type=str, default="./result",
                    help="The path of inference result bin file.")
parser.add_argument("--img_conf_path", type=str, default="./data/img_info",
                    help="The path of img config path, include the image name ,width and height.")
                    
args = parser.parse_args()

def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #print(areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label



def get_default_dict():
    return {"image_id": -1, "category_id": -1, "bbox": [], "score": 0}


def parse_img_config():
    img_info = args.img_info_path
    with open(img_info, 'r')as f:
        img_info_list = f.read().split('\n')[:-1]
        print(len(img_info_list))
    img_info_dict = {}
    count = 0
    for i in img_info_list:
        tmp_list = i.split(' ')
        idx = int(count)
        imgName = tmp_list[0][tmp_list[0].rfind('/') + 1:]
        print("==============")
        print(imgName)
        scale_w = float(tmp_list[1])
        scale_h = float(tmp_list[2])
        left = int(tmp_list[3])
        top = int(tmp_list[4])
        bbox_len = 4
        bbox = []
        img_info_dict[idx] = {
            'imgName' : imgName,
            'scale_w' : scale_w,
            'scale_h': scale_h,
            'left': left,
            'top': top
        }
        count = count + 1
    return img_info_dict

def process(img_info_dict, real_id_to_cat_id, json_out):
    test_len = len(img_info_dict.keys())
    for test_idx in range(test_len):
        img_name = img_info_dict[test_idx]['imgName']
        img_temp = img_name[0:img_name.rindex('.')]
        scores_ = np.fromfile(('{}/davinci_{}_output0.bin').format(args.result_file_path, img_temp), dtype='float32').reshape(1,10647,80)
        boxes_ = np.fromfile(('{}/davinci_{}_output1.bin').format(args.result_file_path, img_temp), dtype='float32').reshape(1,10647,4)
        boxes_, scores_, labels_ = cpu_nms(boxes_, scores_, args.num_class, args.max_boxes, args.score_thresh, args.nms_thresh)
            
        resize_ratio_w = img_info_dict[test_idx]['scale_w']
        resize_ratio_h = img_info_dict[test_idx]['scale_h']
        dw = img_info_dict[test_idx]['left']
        dh = img_info_dict[test_idx]['top']
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio_w
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio_h

        img_temp = img_temp.split('.')
        if args.save_json:
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                bw = x1 - x0
                bh = y1 - y0
                s = scores_[i]
                c = labels_[i]
                t_dict = get_default_dict()
                t_dict['image_id'] = int(img_temp[0][img_temp[0].rindex('_') + 1: ])
                t_dict['category_id'] = real_id_to_cat_id[int(c) + 1]
                t_dict['bbox'] = [int(i) for i in [x0, y0, bw, bh]]
                t_dict['score'] = float(s)
                json_out.append(t_dict)

def get_resize_config(new_width, new_height):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    img_conf = args.img_conf_path
    with open(img_conf, 'r')as f:
        img_info_list = f.read().split('\n')[:-1]
        print(len(img_info_list))
    img_info_dict = {}
    count = 0
    for i in img_info_list:
        tmp_list = i.split(' ')
        idx = int(tmp_list[0])
        imgName = tmp_list[1][tmp_list[1].rfind('/') + 1:]
        #print(imgName)
        ori_width = int(tmp_list[2])
        ori_height = int(tmp_list[3])
        resize_ratio = min(new_width / ori_width, new_height / ori_height)
        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)
        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)
        img_info_dict[idx] = {
            'imgName' : imgName,
            'scale_w' : resize_ratio,
            'scale_h': resize_ratio,
            'left': dw,
            'top': dh
        }
    return img_info_dict


def post_process():
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)

    cat_id_to_real_id = \
        {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
         18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
         35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
         50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
         64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
         82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
    real_id_to_cat_id = {cat_id_to_real_id[i]: i for i in cat_id_to_real_id}
    if args.img_info_path != "":
        img_info_dict = parse_img_config()
    else:
        img_info_dict = get_resize_config(416, 416)

    json_out = []
    process(img_info_dict, real_id_to_cat_id, json_out)
    if args.save_json:
        with open(args.save_json_path, 'w')as f:
            json.dump(json_out, f)
        #print('output json saved to: ', args.save_json_path)


if __name__ == '__main__':
    post_process()
    #os.system('python3.7 eval_coco.py %s' % args.save_json_path)
