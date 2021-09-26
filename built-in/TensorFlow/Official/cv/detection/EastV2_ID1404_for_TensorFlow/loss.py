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

from npu_bridge.npu_init import *
import tensorflow as tf
from config import batch_size_per_gpu as BATCH_SIZE_PER_GPU


def dice_loss(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def ohem_single(gt_text, score, mask):
    """
    Online hard negative mining.
    Return:
        the mask of (selected negative samples + all positve samples).
        if n_pos == 0,then top 10,000 negative samples will be selected.
        if n_pos ~=0, then 3*n_pos negative samples are selected. 
    """
    # _, h, w, _ = gt_text.shape.as_list()
    tt_pos_num = tf.reduce_sum(gt_text)

    pos_num = tt_pos_num - tf.reduce_sum(gt_text * mask)
    neg_num = tf.cond(pos_num > 0, lambda: pos_num * 3, lambda: tf.constant(10000.))
    max_neg_entries = tf.reduce_sum(1. - gt_text)
    neg_num = tf.minimum(neg_num, max_neg_entries)
    neg_conf = tf.boolean_mask(score, gt_text < 0.5)
    vals, _ = tf.nn.top_k(neg_conf, k=tf.cast(neg_num, dtype=tf.int32))
    thresh = vals[-1]

    selected_mask = tf.logical_or(gt_text > 0.5, score >= thresh)
    selected_mask = tf.logical_and(selected_mask, mask > 0.5)
    return selected_mask


def ohem_batch(y_true_cls, y_pred_cls, training_mask):
    """
    Online hard examples mining for batch.
    """
    N = BATCH_SIZE_PER_GPU
    selected_masks = []
    for idx in range(N):
        gt_text = y_true_cls[idx, :, :, :]
        score = y_pred_cls[idx, :, :, :]
        mask = training_mask[idx, :, :, :]
        selected_masks.append(ohem_single(gt_text, score, mask))
    selected_masks = tf.stack(selected_masks, axis=0)
    return tf.cast(selected_masks, dtype=tf.float32)


def weighted_bce_general(y_true_cls, y_pred_cls, training_mask):
    num_mask = tf.reduce_sum(training_mask)
    beta = tf.reduce_sum(y_true_cls * training_mask) / (num_mask + 0.01)
    labels = tf.multiply(y_true_cls, training_mask)
    predicts = tf.multiply(y_pred_cls, training_mask)
    eps = 1e-5
    score_loss = - tf.reduce_sum(
        (1. - beta) * labels * tf.log(predicts + eps) + beta * (1 - labels) * tf.log(1 - predicts + eps)
    ) / (num_mask + 0.01)
    return score_loss


def weighted_bce(y_true_cls, y_pred_cls, training_mask):
    beta = 1 - tf.reduce_mean(tf.multiply(y_true_cls, training_mask))
    labels = tf.multiply(y_true_cls, training_mask)
    predicts = tf.multiply(y_pred_cls, training_mask)
    # log +epsilon for stable cal
    score_loss = - tf.reduce_mean(
        beta * labels * tf.log(predicts + 1e-5) + (1 - beta) * (1 - labels) * tf.log(1 - predicts + 1e-5)
    )
    return score_loss


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask, name='level'):
    """
    define the loss used for training, contraning two part,
    First part: Dice loss + BCE loss 
    Second part: IOU loss + angle loss 
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    """
    # OHEM, dice-loss and bce-loss
    selected_mask = ohem_batch(y_true_cls, y_pred_cls, training_mask)
    dice_cls_loss = dice_loss(y_true_cls, y_pred_cls, selected_mask)
    bce_cls_loss = weighted_bce_general(y_true_cls, y_pred_cls, selected_mask)

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 0.1) / (area_union + 0.1))
    L_theta = 1. - tf.cos(theta_pred - theta_gt)

    aabb_loss = tf.reduce_mean(L_AABB * y_true_cls)
    theta_loss = tf.reduce_mean(L_theta * y_true_cls)

    tf.summary.scalar('{}/geometry_AABB'.format(name), aabb_loss)
    tf.summary.scalar('{}/geometry_theta'.format(name), theta_loss)
    tf.summary.scalar('{}/text_bce_loss'.format(name), bce_cls_loss)
    tf.summary.scalar('{}/text_dice_loss'.format(name), dice_cls_loss)

    # scale classification loss to match the iou loss part
    return (aabb_loss + 20. * theta_loss)*10 + (dice_cls_loss + bce_cls_loss)
