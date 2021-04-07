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

from resnet import resnet_v2_50, resnet_arg_scope
import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import conv, merge, relu
from config import *


def class_subnet(inputs):
    with tf.variable_scope("class_subnet", reuse=tf.AUTO_REUSE):
        inputs = relu(conv("conv1", inputs, 256, 3, 1, "SAME"))
        inputs = relu(conv("conv2", inputs, 256, 3, 1, "SAME"))
        inputs = relu(conv("conv3", inputs, 256, 3, 1, "SAME"))
        inputs = relu(conv("conv4", inputs, 256, 3, 1, "SAME"))
        inputs = conv("conv5", inputs, K*A, 3, 1, "SAME", True)
        H, W = tf.shape(inputs)[1], tf.shape(inputs)[2]
        inputs = tf.reshape(inputs, [-1, H * W * A, K])
    return inputs

def box_subnet(inputs):
    with tf.variable_scope("box_subnet", reuse=tf.AUTO_REUSE):
        inputs = relu(conv("conv1", inputs, 256, 3, 1, "SAME"))
        inputs = relu(conv("conv2", inputs, 256, 3, 1, "SAME"))
        inputs = relu(conv("conv3", inputs, 256, 3, 1, "SAME"))
        inputs = relu(conv("conv4", inputs, 256, 3, 1, "SAME"))
        inputs = conv("conv5", inputs, 4*A, 3, 1, "SAME")
        H, W = tf.shape(inputs)[1], tf.shape(inputs)[2]
        inputs = tf.reshape(inputs, [-1, H * W * A, 4])
    return inputs

def backbone(inputs, is_training):
    arg_scope = resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        _, end_points = resnet_v2_50(inputs, is_training=is_training)
    C3 = end_points["resnet_v2_50/block2/unit_3/bottleneck_v2"]
    C4 = end_points["resnet_v2_50/block3/unit_5/bottleneck_v2"]
    C5 = end_points["resnet_v2_50/block4/unit_3/bottleneck_v2"]
    P5 = conv("conv5", C5, 256, 1, 1, "SAME")
    P4 = merge("merge1", C4, P5)
    P3 = merge("merge2", C3, P4)
    P6 = conv("conv6", C5, 256, 3, 2, "SAME")
    P7 = conv("conv7", relu(P6), 256, 3, 2, "SAME")

    P3_class_logits = class_subnet(P3)
    P3_box_logits = box_subnet(P3)

    P4_class_logits = class_subnet(P4)
    P4_box_logits = box_subnet(P4)

    P5_class_logits = class_subnet(P5)
    P5_box_logits = box_subnet(P5)

    P6_class_logits = class_subnet(P6)
    P6_box_logits = box_subnet(P6)

    P7_class_logits = class_subnet(P7)
    P7_box_logits = box_subnet(P7)
    class_logits = tf.concat([P3_class_logits, P4_class_logits, P5_class_logits, P6_class_logits, P7_class_logits], axis=1)
    box_logits = tf.concat([P3_box_logits, P4_box_logits, P5_box_logits, P6_box_logits, P7_box_logits], axis=1)
    class_logits_dict = {"P3": P3_class_logits, "P4": P4_class_logits, "P5": P5_class_logits,
                         "P6": P6_class_logits, "P7": P7_class_logits}
    box_logits_dict = {"P3": P3_box_logits, "P4": P4_box_logits, "P5": P5_box_logits,
                       "P6": P6_box_logits, "P7": P7_box_logits}
    return class_logits, box_logits, class_logits_dict, box_logits_dict
    pass

# inputs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
# is_training = tf.placeholder(tf.bool)
# backbone(inputs, is_training)