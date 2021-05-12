# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# ============================================================================
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
# coding: utf-8
# This script is used to remove the optimizer parameters in the saved checkpoint files.
# These parameters are useless in the forward process. 
# Removing them will shrink the checkpoint size a lot.

import sys
sys.path.append('..')

import os
import tensorflow as tf
from model import yolov3

# params
ckpt_path = ''
class_num = 20
save_dir = 'shrinked_ckpt'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image = tf.placeholder(tf.float32, [1, 416, 416, 3])
yolo_model = yolov3(class_num, None)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image)

saver_to_restore = tf.train.Saver()
saver_to_save = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_to_restore.restore(sess, ckpt_path)
    saver_to_save.save(sess, save_dir + '/shrinked')
