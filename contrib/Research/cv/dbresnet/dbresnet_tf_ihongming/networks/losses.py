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
import tensorflow as tf
import tensorflow.keras as K


def balanced_crossentropy_loss(pred, gt, mask, negative_ratio=3.):
    positive_mask = (gt * mask)
    negative_mask = ((1 - gt) * mask)
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    loss = K.backend.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    # negative_loss, _ = tf.math.top_k(tf.reshape(negative_loss, (3276800,)), 10)
    negative_loss, _ = tf.math.top_k(tf.reshape(negative_loss, (-1,)), 300000)

    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    return balanced_loss, loss


def dice_loss(pred, gt, mask, weights):
    # pred = pred[..., 0]
    intersection = tf.reduce_sum(pred * gt * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union
    return loss


def l1_loss(pred, gt, mask):
    # pred = pred[..., 0]
    mask_sum = tf.reduce_sum(mask)
    loss = K.backend.switch(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / (mask_sum + 1e-6), tf.constant(0.))
    return loss


def db_loss(binarize_map, threshold_map, thresh_binary,
            gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask, alpha=5.0, beta=10.0, ohem_ratio=3.0):
    threshold_loss = l1_loss(threshold_map, gt_threshold_map, gt_thresh_mask)
    binarize_loss, _ = balanced_crossentropy_loss(binarize_map, gt_score_maps, gt_score_mask, negative_ratio=ohem_ratio)
    thresh_binary_loss = dice_loss(thresh_binary, gt_score_maps, gt_score_mask, _)

    model_loss = alpha * binarize_loss + beta * threshold_loss + thresh_binary_loss
    tf.summary.scalar('losses/binarize_loss', binarize_loss)
    tf.summary.scalar('losses/threshold_loss', threshold_loss)
    tf.summary.scalar('losses/thresh_binary_loss', thresh_binary_loss)
    return model_loss


def _compute_cls_acc(pred, gt, mask):
    zero = tf.zeros_like(pred, tf.float32)
    one = tf.ones_like(pred, tf.float32)

    pred = tf.where(pred < 0.3, x=zero, y=one)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred * mask, gt * mask), tf.float32))
    return acc


def db_acc(binarize_map, threshold_map, thresh_binary,
           gt_score_maps, gt_threshold_map, gt_score_mask, gt_thresh_mask):
    binarize_acc = _compute_cls_acc(binarize_map, gt_score_maps, gt_score_mask)
    thresh_binary_acc = _compute_cls_acc(thresh_binary, gt_score_maps, gt_score_mask)
    tf.summary.scalar('acc/binarize_acc', binarize_acc)
    tf.summary.scalar('acc/thresh_binary_acc', thresh_binary_acc)
    return binarize_acc, thresh_binary_acc
