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
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
import tensorflow as tf

import common.utils as utils


def alpha_loss(alpha_scores, masks_float32):
    return tf.losses.absolute_difference(alpha_scores, masks_float32)


def comp_loss(alpha_scores, masks_float32, images):
    normalized_images = images / 255

    reconst_fg = tf.multiply(normalized_images, alpha_scores)
    true_fg = tf.multiply(normalized_images, masks_float32)

    return tf.losses.absolute_difference(reconst_fg, true_fg)


def grad_loss(alpha_scores, masks_float32):
    filter_x = tf.constant([[-1/8, 0, 1/8],
                            [-2/8, 0, 2/8],
                            [-1/8, 0, 1/8]],
                           name="sober_x", dtype=tf.float32, shape=[3, 3, 1, 1])
    filter_y = tf.constant([[-1/8, -2/8, -1/8],
                            [0, 0, 0],
                            [1/8, 2/8, 1/8]],
                           name="sober_y", dtype=tf.float32, shape=[3, 3, 1, 1])
    filter_xy = tf.concat([filter_x, filter_y], axis=-1)

    grad_alpha = tf.nn.conv2d(alpha_scores, filter_xy, strides=[1, 1, 1, 1], padding="SAME")
    grad_masks = tf.nn.conv2d(masks_float32, filter_xy, strides=[1, 1, 1, 1], padding="SAME")

    return tf.losses.absolute_difference(grad_alpha, grad_masks)


def kd_loss(scores, masks):
    logits = tf.reshape(scores, [-1, 2])  # (?, 2)
    masks_foreground = tf.reshape(masks, [-1])  # foreground goes to value 1
    labels = tf.stack([1 - masks_foreground, masks_foreground], axis=1)
    loss = softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=tf.stop_gradient(labels),
    )

    cost = tf.reduce_mean(loss)
    return cost
