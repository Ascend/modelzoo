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
from tensorflow.contrib import slim
from model.resnet import resnet_v1


def res_down(feature, is_training):
    feature = conv2d_with_batch_norm(feature, filter=256, kernel=1, padding='valid', is_training=is_training)
    b, h, w, c = feature.shape
    l = 4
    r = -4
    if h < 20:
        feature = feature[:, l:r, l:r, :]
    return feature


def conv2d_with_batch_norm(input_, filter, kernel, stride=1, padding='valid',
                           dilation=1, use_bias=False, is_training=None):
    """Conv2d layer with batch normalisation."""
    out = tf.layers.conv2d(input_, filters=filter, kernel_size=kernel,
                           strides=stride, padding=padding,
                           dilation_rate=dilation, use_bias=use_bias, kernel_initializer=tf.initializers.he_normal())
    out = tf.layers.batch_normalization(out, momentum=0.9, epsilon=1e-5, training=is_training)
    return out


def adjust(input_, scope, in_channels, hidden, out_channels, kernel_size=3, is_training=None):
    out = conv2d_with_batch_norm(input_, filter=hidden, kernel=kernel_size, padding='valid', is_training=is_training)
    out = tf.nn.relu(out)
    return out


def conv2d_dw_group(input_, kernel):
    batch_size, i_h, i_w, i_c = input_.shape
    _, k_h, k_w, k_c = kernel.shape
    input_ = tf.transpose(input_, [0, 3, 1, 2])
    kernel = tf.transpose(kernel, [0, 3, 1, 2])
    input_ = tf.reshape(input_, [1, batch_size * i_c, i_h, i_w])
    kernel = tf.reshape(kernel, [batch_size * k_c, 1, k_h, k_w])

    input_ = tf.transpose(input_, [0, 2, 3, 1])
    kernel = tf.transpose(kernel, [2, 3, 0, 1])

    out = tf.nn.depthwise_conv2d(input_, kernel, [1, 1, 1, 1], 'VALID')
    out = tf.reshape(out, [out.shape[1], out.shape[2], batch_size, k_c])
    out = tf.transpose(out, [2, 0, 1, 3])
    return out


def depth_corr(template_feature, search_feature, in_channels, hidden, out_channels, kernel_size, is_training):
    kernel = adjust(template_feature, 'Exemplar', in_channels, hidden, out_channels, kernel_size,
                    is_training=is_training)
    input = adjust(search_feature, 'Search', in_channels, hidden, out_channels, kernel_size,
                   is_training=is_training)

    feature = conv2d_dw_group(input, kernel)

    input_ = adjust(feature, 'DepthCorr_head', hidden, hidden, out_channels, 1, is_training=is_training)
    out = tf.layers.conv2d(input_, filters=out_channels, kernel_size=1,
                           strides=1, kernel_initializer=tf.initializers.he_normal())
    return out


def softmax(cls):
    b, h, w, a2 = cls.shape
    cls = tf.reshape(cls, (b, h, w, 2, a2 // 2))
    cls = tf.transpose(cls, [0, 1, 2, 4, 3])
    # cls = tf.nn.log_softmax(cls, axis=4)
    return cls


def resnet_feature(template, search, is_training):
    with tf.name_scope("resnet_feature"):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, end_points = resnet_v1.resnet_v1_50(inputs=template, is_training=is_training, scope="siammask_resnet")
            c2, c3, c4, c5 = end_points['pool2'], end_points['pool3'], end_points['pool4'], end_points['pool5']
            template_feature = c5

            _, end_points2 = resnet_v1.resnet_v1_50(inputs=search, is_training=is_training, scope="siammask_resnet",
                                                    reuse=True)
            c_2_2, c_2_3, c_2_4, c_2_5 = end_points2['pool2'], end_points2['pool3'], \
                                         end_points2['pool4'], end_points2['pool5']
            search_feature = c_2_5
            template_feature = res_down(template_feature, is_training)
            search_feature = res_down(search_feature, is_training)
    return template_feature, search_feature


def rpn_model(template_feature, search_feature, is_training):
    with tf.name_scope("rpn_model"):
        rpn_pred_cls = depth_corr(template_feature, search_feature, 256, 256, 10, 3, is_training)
        rpn_pred_loc = depth_corr(template_feature, search_feature, 256, 256, 20, 3, is_training)
    return rpn_pred_cls, rpn_pred_loc


def mask_model(template_feature, search_feature, is_training):
    with tf.name_scope("mask_model"):
        oSz = 63
        rpn_pred_mask = depth_corr(template_feature, search_feature, 256, 256, oSz ** 2, 3, is_training)
    return rpn_pred_mask


def siammask(template, search, is_training=True):
    with tf.name_scope('siammask'):
        template_feature, search_feature = resnet_feature(template, search, is_training=is_training)
        rpn_pred_cls, rpn_pred_loc = rpn_model(template_feature, search_feature, is_training=is_training)
        rpn_pred_mask = mask_model(template_feature, search_feature, is_training=is_training)
        if is_training:
            rpn_pred_cls = softmax(rpn_pred_cls)
    return rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature
