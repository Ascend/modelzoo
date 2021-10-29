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

from script.utils import postprocess

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane', 5: 'bus', 
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

class yolov2_npu:
 
    def run(self, npu_output):
        bboxes = np.fromfile(os.path.join(npu_PATH,npu_output[:-5]+"2.bin"), dtype="float32").reshape(1, 169, 5, 4)
        obj_probs = np.fromfile(os.path.join(npu_PATH,npu_output), dtype="float32").reshape(1, 169, 5)
        class_probs = np.fromfile(os.path.join(npu_PATH,npu_output[:-5]+"1.bin"), dtype="float32").reshape(1, 169, 5, 80)
        
        image = cv2.imread(sys.argv[2] + npu_output.split("_output")[0].split("davinci_")[1] + ".jpg")
        image_shape = image.shape[:2]
                
        boxes, scores, classes = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)
        path = './detections_npu/'
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(os.path.join(path + npu_output.split("_output")[0].split("davinci_")[1] + ".txt"),'w') as f:
            for i in range(len(scores)):
                if ' ' in labels_to_names[classes[i]]:
                    labels_to_name = labels_to_names[classes[i]].split(' ')[0] + labels_to_names[classes[i]].split(' ')[1]
                    f.write(labels_to_name + " " + str(scores[i]) + " " + str(boxes[i][0])+ " " + str(boxes[i][1])+ " " + str(boxes[i][2])+ " " + str(boxes[i][3])+'\n')
                else:
                    f.write(labels_to_names[classes[i]] + " " + str(scores[i]) + " " + str(boxes[i][0])+ " " + str(boxes[i][1])+ " " + str(boxes[i][2])+ " " + str(boxes[i][3])+'\n')

if __name__ == "__main__":
    all_result_NAME = os.listdir(npu_PATH)
    all_result_NAME.sort()
    all_image_NAME = [fn for fn in all_result_NAME if fn[-5]=="0"]
    
    yolov2_Npu = yolov2_npu()
    for npu_output in all_image_NAME:
        yolov2_Npu.run(npu_output)
