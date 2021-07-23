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

import tensorflow as tf
tf.disable_eager_execution()
import numpy as np
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.platform import gfile
import cv2
import sys

SCORETHRES = 0.2
INPUTSIZE = 416
IOU = 0.45
SCORE = 0.25

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
                    6: 'train', 7: 'truck', 8: 'boat', 9: 'trafficlight', 10: 'firehydrant', 
                    11: 'stopsign', 12: 'parkingmeter', 13: 'bench', 14: 'bird', 15: 'cat', 
                    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 
                    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 
                    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 
                    31: 'snowboard', 32: 'sportsball', 33: 'kite', 34: 'baseballbat', 
                    35: 'baseballglove', 36: 'skateboard', 37: 'surfboard', 38: 'tennisracket', 39: 'bottle', 
                    40: 'wineglass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
                    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
                    51: 'carrot', 52: 'hotdog', 53: 'pizza', 54: 'donut', 55: 'cake', 
                    56: 'chair', 57: 'couch', 58: 'pottedplant', 59: 'bed', 60: 'diningtable', 
                    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
                    66: 'keyboard', 67: 'cellphone', 68: 'microwave', 69: 'oven', 
                    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
                    75: 'vase', 76: 'scissors', 77: 'teddybear', 78: 'hairdrier', 79: 'toothbrush'}

npu_PATH = sys.argv[1]
original_PATH = sys.argv[2]

def writeResult(path, filename, b, s, l):
    pic_name = filename.split("davinci_")[1].split("_output")[0]
    file_to_be_write = open(os.path.join(path, pic_name+".txt"), "w")
    original_image = cv2.imread(os.path.join(original_PATH, pic_name+".jpg"))
    print(os.path.join(original_PATH, pic_name+".jpg"))
    h, w, c = original_image.shape
    for box, score, label in zip(b[0], s[0], l[0]):
        if score < 0.25:
            break
        file_to_be_write.write(labels_to_names[label]+" "+str(score)+" "+str(int(box[1]*w))+" "+str(int(box[0]*h))+" "+str(int(box[3]*w-box[1]*w))+" "+str(int(box[2]*h-box[0]*h))+"\n")
    file_to_be_write.close()

class yolov4_npu:
    def __init__(self, score_threshold=SCORETHRES, input_shape=tf.constant([INPUTSIZE,INPUTSIZE])):
        self.sess = tf.Session()
        self.bbox = tf.placeholder(dtype="float32", shape=[1,10647,4], name="pred_bbox")
        self.prob = tf.placeholder(dtype="float32", shape=[1, 10647, 80], name="pred_prob")
        scores_max = tf.math.reduce_max(self.prob, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tf.boolean_mask(self.bbox, mask)
        pred_conf = tf.boolean_mask(self.prob, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(self.bbox)[0], -1, tf.shape(class_boxes)[-1]])
        pred_conf = tf.reshape(pred_conf, [tf.shape(self.prob)[0], -1, tf.shape(pred_conf)[-1]])

        box_xy, box_wh = tf.split(class_boxes, (2,2), axis=-1)

        input_shape = tf.cast(input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw/2.))/input_shape
        box_maxes = (box_yx + (box_hw/2.))/input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  #y_min
            box_mins[..., 1:2],  #x_min
            box_maxes[..., 0:1], #y_max
            box_maxes[..., 1:2]  #x_max
        ], axis=-1)
        pred_bbox = tf.concat([boxes, pred_conf], axis=-1)
        boxes = pred_bbox[:, :, 0:4]
        pred_conf = pred_bbox[:, :, 4:]
        self.output = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU,
            score_threshold=SCORE
        )

    def run(self, npu_output):
        pred_box = np.fromfile(os.path.join(npu_PATH, npu_output[:-5]+"1.bin"), dtype="float32").reshape(1, 10647, 4)
        pred_prob = np.fromfile(os.path.join(npu_PATH, npu_output), dtype="float32").reshape(1, 10647, 80)
        boxes, scores, classes, valid_detections = self.sess.run(self.output, feed_dict={self.bbox:pred_box, self.prob:pred_prob})
        writeResult("./detections_npu/", npu_output, boxes, scores, classes)

if __name__ == "__main__":
    all_result_NAME = os.listdir(npu_PATH)
    all_result_NAME.sort()
    all_image_NAME = [fn for fn in all_result_NAME if fn[-5]=="0"]
    output_dir = "./detections_npu/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    print(len(all_image_NAME))
    yolov4_Npu = yolov4_npu()
    for npu_output in all_image_NAME:
        yolov4_Npu.run(npu_output)
