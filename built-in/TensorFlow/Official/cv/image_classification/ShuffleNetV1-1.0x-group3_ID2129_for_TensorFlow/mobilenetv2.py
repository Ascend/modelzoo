#
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
#
from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def mobilenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of MobileNet v2:
    https://arxiv.org/pdf/1801.04381.pdf

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
    """

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x
    if False:
        with tf.name_scope('image_preprocess'):
            #0:bilinear,1:NEAREST,2:cubic,3:area
            images=tf.image.resize_images(images,[224,224],method=0)
            images=(1.0 / 255.0) * tf.to_float(images)
            images=tf.reshape(images, (1,224,224,1))
    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    expansion = 6
    mutilplier = 1
    blockUnits = [1,2,3,4,3,3,1]
    blockOutputChannels = [16,24,32,64,96,160,320]
    strides=[1,2,2,2,1,2,1]
    for i in range(len(blockUnits)):
        blockOutputChannels[i] = int(blockOutputChannels[i]*mutilplier)
        
    with tf.variable_scope('MobileNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer(),
            'biases_initializer': None
        }

        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            x = slim.conv2d(x, 32, (3, 3), stride=2, scope='Conv1')
            #x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            for blocki in range(7):
                x = block(x, blockOutputChannels[blocki],strides[blocki],blockUnits[blocki],exp=expansion,scope='block'+str(blocki))

            x = slim.conv2d(x, 1280, (1, 1), stride=1, scope='Conv9')
    
    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])
    x = tf.reshape(x, [-1,1,1,1280])
    logits = slim.conv2d(x, num_classes, (1, 1), stride=1, biases_initializer=None, scope='Conv10')
    logits = tf.reshape(logits,[-1,num_classes])
    #logits = slim.fully_connected(
     #   x, num_classes, activation_fn=None, scope='classifier',
      #  weights_initializer=tf.contrib.layers.xavier_initializer()
    #)
    return logits


def bottleneck(x, out_channels=None, stride=1, exp=1, scope='bottleneck'):
    with tf.variable_scope(scope):
        #shape = tf.shape(x)
        ch = x.shape[3].value
        y=slim.conv2d(x,ch*exp,(1,1),scope='conv1x1_before')
        y=depthwise_conv(y,3,stride=stride,activation_fn=tf.nn.relu, scope='depthwise_conv')
        y=slim.conv2d(y,out_channels,(1,1),activation_fn=None, scope='conv1x1_after')
        if stride == 1 and ch != out_channels:
            x=slim.conv2d(x,out_channels,(1,1),scope='shorcut')
            y=x+y
    return y


def block(x, out_channels=None, stride=1, num_units=1, exp=1, scope='block'):
    with tf.variable_scope(scope):
        x = bottleneck(x, out_channels, stride=stride, exp=exp, scope='bottleneck0')
        for i in range(1, num_units):
            x = bottleneck(x, out_channels, exp=exp, scope='bottleneck'+str(i))
    return x


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=None,
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x

