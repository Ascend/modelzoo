# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf

from npu_bridge.estimator import npu_ops

# vgg with initialization method in gluoncv
def vgg_impl(inputs, is_training=True, class_num=1000):
    x = inputs

    # conv1
    x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp1
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # covn2
    x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp2
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # conv3
    x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp3
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # conv4
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp4
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    # conv5
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))
    x = tf.layers.conv2d(x, 512, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', use_bias=True, kernel_initializer=tf.initializers.variance_scaling(scale=2.0, mode='fan_out'))

    # mp5
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='SAME')

    x = tf.reshape(x, [-1, 7 * 7 * 512])

    # fc6
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    # drop6
    if is_training:
        x = npu_ops.dropout(x, 0.5)
    # fc7
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    # drop7
    if is_training:
        x = npu_ops.dropout(x, 0.5)
    # fc8
    x = tf.layers.dense(x, class_num, activation=None, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    return x

