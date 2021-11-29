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
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights

num_class = 80
img_size = 416
weight_path = './data/darknet_weights/darknet53.conv.74'
save_path = './data/darknet_weights/darknet53.ckpt'
anchors = parse_anchors('./data/yolo_anchors.txt')

model = yolov3(num_class, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)

    sess.run(tf.global_variables_initializer())
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



