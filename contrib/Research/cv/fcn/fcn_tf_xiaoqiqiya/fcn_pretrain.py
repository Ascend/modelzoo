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
import tensorflow.contrib.slim as slim


def vgg_arg_scope(weight_decay=0.0005,
                  use_batch_norm=True,
                  batch_norm_decay=0.9997,
                  batch_norm_epsilon=0.001,
                  batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                  batch_norm_scale=False):
    """Defines the VGG arg scope.
    Args:
      weight_decay: The l2 regularization coefficient.
    Returns:
      An arg_scope.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': batch_norm_epsilon,
        # collection containing update_ops.
        'updates_collections': batch_norm_updates_collections,
        # use fused batch norm if possible.
        'fused': None,
        'scale': batch_norm_scale,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params) as arg_sc:
            return arg_sc


def fcn8s(inputs,
          is_training,
          num_classes=21,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_16'):
    # pretrain encoder part
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        # vgg encoder part
        with tf.variable_scope(scope, 'vgg_16', [inputs]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            #print("--------net1--------", net)
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            #print("--------net2--------", net)
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            shorcut2 = net
            #print("--------net3--------", net)
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            shorcut3 = net
            #print("--------net4-------", net)
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # decoder part
        with tf.variable_scope('upsample'):
            net = slim.conv2d(net, 4096, [7, 7], 1, padding="same")
            net = slim.dropout(net,0.5)
            net = slim.conv2d(net, 4096, [1, 1], 1, padding="same")
            net = slim.dropout(net,0.5)
            net = tf.layers.conv2d(net, 21, [1, 1], 1, padding="same")
            #print("--------out--------", net)
            #netup1 = tf.layers.conv2d_transpose(net, 512, 4, 2, padding="same")
            #netup1 = netup1 + shorcut3
            netup1 = tf.layers.conv2d_transpose(net, 512, 4, 2, padding="valid")
            netup1 = netup1[:,1:-1,1:-1,:] + shorcut3

            #print("--------netup1--------", netup1)
            # upstage2
            #netup2 = tf.layers.conv2d_transpose(netup1, 256, 4, 2,padding="same")
            #netup2 = netup2 + shorcut2
            netup2 = tf.layers.conv2d_transpose(netup1, 256, 4, 2,padding="valid")
            netup2 = netup2[:,1:-1,1:-1,:] + shorcut2

            #print("--------netup2--------", netup2)
            # upstage3
            #netup3 = tf.layers.conv2d_transpose(netup2, 21, 16, 8, use_bias=False, padding="same")
            netup3 = tf.layers.conv2d_transpose(netup2, 21, 16, 8, use_bias=False, padding="valid")
            #print("--------netup3--------", netup3)
            return netup3[:,4:-4,4:-4,:]
