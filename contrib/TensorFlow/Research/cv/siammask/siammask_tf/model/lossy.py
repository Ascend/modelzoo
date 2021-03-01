# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
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
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# limitations under the License.
# ============================================================================
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

# -------------------------------------------------------------------------------

import tensorflow as tf


def get_cls_loss(pred, label, select):
    if tf.size(select) == 0:
        return tf.reduce_sum(pred) * 0
    # label has value -1, which will make log(-1) become nan, so use  tf.abs(label) make -1 to 1,
    # sparse_logits * select will make -1 has no use
    label = tf.abs(label)
    sparse_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(label, tf.int32), logits=pred)
    select = tf.cast(select, dtype=tf.float32)
    loss = sparse_logits * select
    loss = tf.reduce_sum(loss) / tf.reduce_sum(select)
    return loss


def select_cross_entropy_loss(pred, label):
    pred = tf.reshape(pred, (-1, 2))
    label = tf.reshape(label, (-1,))
    weight = tf.math.equal(tf.cast(label, dtype=tf.int32), tf.constant(1))
    pos = tf.cast(weight, dtype=tf.int32)

    weight = tf.math.equal(tf.cast(label, dtype=tf.int32), tf.constant(0))
    neg = tf.cast(weight, dtype=tf.int32)

    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, sh, sw, _ = pred_loc.shape
    pred_loc = tf.reshape(pred_loc, (b, sh, sw, 4, -1))
    diff = tf.abs(pred_loc - label_loc)
    diff = tf.reshape(tf.reduce_sum(diff, axis=3), (b, sh, sw, -1))
    loss = diff * loss_weight
    return tf.reduce_sum(loss) / tf.cast(b, tf.float32)


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    where = tf.math.equal(tf.cast(tf.reshape(weight, (-1,)), dtype=tf.int32), 1)
    pos = tf.cast(where, dtype=tf.int32)
    p_m = tf.reshape(p_m, (-1, o_sz, o_sz, 1))
    p_m = tf.image.resize_images(p_m, [g_sz, g_sz], align_corners=True)
    p_m = tf.reshape(p_m, (-1, g_sz * g_sz))
    mask_pad = tf.pad(mask, paddings=[[0, 0], [32, 32], [32, 32], [0, 0]], mode="CONSTANT")
    mask_uf = tf.image.extract_image_patches(mask_pad, ksizes=[1, g_sz, g_sz, 1],
                                             strides=[1, 8, 8, 1],
                                             rates=[1, 1, 1, 1],
                                             padding='VALID')
    mask_uf = tf.reshape(mask_uf, (-1, g_sz * g_sz))
    loss = tf.math.softplus(-p_m * mask_uf)
    loss = tf.reduce_sum(loss * tf.reshape(tf.cast(pos, dtype=tf.float32), (-1, 1)), axis=1) / (tf.reduce_sum(
        tf.cast(pos, dtype=tf.float32)) + 1e-7)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf, pos)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label, pos):
    pred = tf.cast(tf.greater_equal(pred, 0), tf.int32) * tf.reshape(pos, (-1, 1))
    mask_sum = tf.add(tf.cast(tf.equal(pred, 1), tf.int32),
                      tf.cast(tf.equal(label, 1), tf.int32)) * tf.reshape(pos, (-1, 1))
    intxn = tf.cast(tf.reduce_sum(tf.cast(tf.equal(mask_sum, 2), tf.int32), axis=-1), tf.float32)
    union = tf.cast(tf.reduce_sum(tf.cast(tf.greater(mask_sum, 0), tf.int32), axis=-1), tf.float32)
    iou = intxn / (union + 1e-7)
    return tf.reduce_sum(iou) / (tf.reduce_sum(tf.cast(pos, dtype=tf.float32)) + 1e-7), \
           tf.cast(tf.reduce_sum(tf.cast(tf.greater(iou, 0.5), tf.int32)), tf.float32) / (tf.reduce_sum(
               tf.cast(pos, dtype=tf.float32)) + 1e-7), \
           tf.cast(tf.reduce_sum(tf.cast(tf.greater(iou, 0.7), tf.int32)), tf.float32) / (tf.reduce_sum(
               tf.cast(pos, dtype=tf.float32)) + 1e-7)


def total_loss(label_cls, label_loc, lable_loc_weight, label_mask, label_mask_weight,
               rpn_pred_cls, rpn_pred_loc, rpn_pred_mask):
    rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)

    rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight)

    rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(rpn_pred_mask, label_mask,
                                                                   label_mask_weight)

    return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7
