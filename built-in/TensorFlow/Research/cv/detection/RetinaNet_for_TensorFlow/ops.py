import tensorflow as tf
import numpy as np


def conv(name, inputs, nums_out, k_size, stride, padding, is_final=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.01))
        if is_final:
            b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([-np.log((1 - 0.01) / 0.01)]))
        else:
            b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, stride, stride, 1], padding)
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def relu(inputs):
    return tf.nn.relu(inputs)

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def merge(name, lateral, top_down):
    with tf.variable_scope(name):
        lateral = conv("conv", lateral, 256, 1, 1, "SAME")
        h, w = tf.shape(lateral)[1], tf.shape(lateral)[2]
        top_down = tf.image.resize_nearest_neighbor(top_down, [h, w])
    return lateral + top_down


def smooth_l1(inputs):
    # x = tf.where(tf.greater(tf.abs(x), 1.0), tf.abs(x) - 0.5, 0.5 * tf.square(x))
    # mask = tf.cast(tf.less(tf.abs(inputs), 1.0), dtype=tf.float32)  # |x| < 1
    # mask_ = tf.abs(mask - 1.0)  # |x| >= 1
    # return 0.5 * tf.square(inputs) * mask + (tf.abs(inputs) - 0.5) * mask_
    loss = tf.where(tf.less(tf.abs(inputs), 1.0), 0.5 * tf.square(inputs), tf.abs(inputs) - 0.5)
    loss = tf.reduce_sum(loss, axis=2)
    return loss


def focal_loss(logits, labels, alpha=0.25, gamma=2):
    pos_pt = tf.clip_by_value(tf.nn.sigmoid(logits), 1e-10, 0.999)
    fl = labels * tf.log(pos_pt) * tf.pow(1 - pos_pt, gamma) * alpha + (1 - labels) * tf.log(1 - pos_pt) * tf.pow(pos_pt, gamma) * (1 - alpha)
    fl = -tf.reduce_sum(fl, axis=2)
    return fl

def offset2bbox(anchors, t_bbox):
    bbox_x = t_bbox[:, 0:1] * anchors[:, 2:3] + anchors[:, 0:1]
    bbox_y = t_bbox[:, 1:2] * anchors[:, 3:4] + anchors[:, 1:2]
    bbox_w = tf.exp(t_bbox[:, 2:3]) * anchors[:, 2:3]
    bbox_h = tf.exp(t_bbox[:, 3:4]) * anchors[:, 3:4]
    x1, y1 = bbox_x - bbox_w / 2, bbox_y - bbox_h / 2
    x2, y2 = bbox_x + bbox_w / 2, bbox_y + bbox_h / 2
    return tf.concat((x1, y1, x2, y2), axis=1)

def top_k_score_bbox(pred_score, pred_bbox, anchors, threshold=0.05, k=1000):
    pred_score_obj = tf.reduce_max(pred_score, axis=1)
    idx = tf.where(tf.greater(pred_score_obj, threshold))[:, 0]
    threshold_score = tf.nn.embedding_lookup(pred_score_obj, idx)
    threshold_bbox = tf.nn.embedding_lookup(pred_bbox, idx)
    threshold_anchors = tf.nn.embedding_lookup(anchors, idx)
    threshold_nums = tf.shape(threshold_score)[0]
    k = tf.where(tf.greater(threshold_nums, k), k, threshold_nums)
    topK_score, topK_indx = tf.nn.top_k(threshold_score, k)
    topK_bbox = tf.nn.embedding_lookup(threshold_bbox, topK_indx)
    topK_anchors = tf.nn.embedding_lookup(threshold_anchors, topK_indx)
    pred_score_idx = tf.nn.embedding_lookup(idx, topK_indx)
    topK_class_score = tf.nn.embedding_lookup(pred_score, pred_score_idx)
    topK_class_labels = tf.argmax(topK_class_score, axis=1)
    return topK_score, topK_bbox, topK_anchors, topK_class_labels


