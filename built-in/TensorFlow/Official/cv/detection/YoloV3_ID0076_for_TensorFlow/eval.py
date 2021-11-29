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

# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms, cpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3
from tqdm import trange
import json
import os,time

# npu modified
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu import util

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
parser.add_argument("--annotation_txt", type=str, default='./data/coco2014_minival.txt',
                    help="The path of the input image. Or annotation label txt.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--max_test", type=int, default=-1,
                    help="max step for test")
parser.add_argument("--score_thresh", type=float, default=1e-3,
                    help="score_threshold for test")
parser.add_argument("--nms_thresh", type=float, default=0.5,
                    help="iou_threshold for test")
parser.add_argument("--max_boxes", type=int, default=100,
                    help="max_boxes for test")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolo3.ckpt",
                    # parser.add_argument("--restore_path", type=str, default="./training_s2/checkpoint_dir/model.ckpt-45800",
                    help="The path of the weights to restore.")
parser.add_argument("--save_img", type=bool, default=False,
                    help="whether to save detected-result image")
parser.add_argument("--save_json", type=bool, default=False,
                    help="whether to save detected-result cocolike json")
parser.add_argument("--save_json_path", type=str, default="./result.json",
                    help="The path of the result.json.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
cat_id_to_real_id = \
    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
     18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
     35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
     50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
     64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
     82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
real_id_to_cat_id = {cat_id_to_real_id[i]: i for i in cat_id_to_real_id}


def get_default_dict():
    return {"image_id": -1, "category_id": -1, "bbox": [], "score": 0}


eval_path = args.annotation_txt
with open(eval_path, 'r')as f:
    eval_file_list = f.read().split('\n')[:-1]
    print(len(eval_file_list))
eval_file_dict = {}
for i in eval_file_list:
    tmp_list = i.split(' ')
    idx = int(tmp_list[0])
    path = tmp_list[1]
    w = float(tmp_list[2])
    h = float(tmp_list[3])
    bbox_len = len(tmp_list[4:]) // 5
    bbox = []
    for bbox_idx in range(bbox_len):
        label, x1, y1, x2, y2 = tmp_list[4:][bbox_idx * 5:bbox_idx * 5 + 5]
        bbox.append([label, x1, y1, x2, y2])
    eval_file_dict[idx] = {
        'path': path,
        'w': w,
        'h': h,
        'bbox': bbox
    }

config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # training on Ascend chips
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

json_out = []
with tf.Session(config=config) as sess:
# with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    # boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=100, score_thresh=args.score_thresh, nms_thresh=0.5)

    saver = tf.train.Saver()
    if args.restore_path.find('.ckpt') < 0 and args.restore_path.find('model-') < 0:
        with open(os.path.join(args.restore_path, 'checkpoint'), 'r')as f:
            tmp_checkpoint = f.readline()
            tmp_checkpoint = tmp_checkpoint.replace('"', '').split(':')[1].strip()
            args.restore_path = os.path.join(args.restore_path, tmp_checkpoint)
            print('tmp_checkpoint: ', tmp_checkpoint)
            # input()

    saver.restore(sess, args.restore_path)

    if args.max_test > 0:
        test_len = min(args.max_test, len(eval_file_dict.keys()))
    else:
        test_len = len(eval_file_dict.keys())
    for test_idx in trange(test_len):
        img_path = eval_file_dict[test_idx]['path']
        img_ori = cv2.imread(img_path)
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        # boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        # print('bbox: ',boxes_)
        t = time.time()
        boxes_, scores_ = sess.run([pred_boxes, pred_scores], feed_dict={input_data: img})
        # print("FPS: ", 1/(time.time() - t))
        boxes_, scores_, labels_ = cpu_nms(boxes_, scores_, args.num_class, args.max_boxes, args.score_thresh, args.nms_thresh)
        # print('bbox: ', boxes_)

        # try:
        #     boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        # except:
        #     print("boxes_: ", boxes_)
        #     continue

        # print("boxes_: ", boxes_)
        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

        if args.save_img:
            # print("box coords:")
            # print(boxes_)
            # print('*' * 30)
            # print("scores:")
            # print(scores_)
            # print('*' * 30)
            # print("labels:")
            # print(labels_)
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1],
                             label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                             color=color_table[labels_[i]])
            cv2.imwrite('tmp/%d_detection_result.jpg' % test_idx, img_ori)
            print('%d done' % test_idx)

        if args.save_json:
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                bw = x1 - x0
                bh = y1 - y0
                s = scores_[i]
                c = labels_[i]
                t_dict = get_default_dict()
                t_dict['image_id'] = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
                t_dict['category_id'] = real_id_to_cat_id[int(c) + 1]
                t_dict['bbox'] = [int(i) for i in [x0, y0, bw, bh]]
                t_dict['score'] = float(s)
                json_out.append(t_dict)

if args.save_json:
    with open(args.save_json_path, 'w')as f:
        json.dump(json_out, f)
    print('output json saved to: ', args.save_json_path)

    os.system('python3.7 eval_coco.py %s'%args.save_json_path)
