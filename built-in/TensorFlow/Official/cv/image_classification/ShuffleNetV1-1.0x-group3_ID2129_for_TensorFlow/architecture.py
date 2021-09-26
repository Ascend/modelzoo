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


def shufflenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
        width_config = {
            0.25: (24, 48, 96, 512),
            0.33: (32, 64, 128, 512),
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (244, 488, 976, 2048),
        }
    """
    possibilities = {'0.33': 32, '0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

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

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv, slim.conv2d_transpose], **params):

            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')
            x = block(x, num_units=8, scope='Stage3')
            '''
            z = slim.conv2d(x, 64, (1, 1), stride=1, scope='Conv6_5')
            z = depthwise_conv(z, kernel=3, stride=2, padding='SAME',scope='Conv6_6')
            z = slim.conv2d(z, 64, (1, 1), stride=1, scope='Conv6_7')
            u = slim.conv2d(x, 64, (1, 1), stride=1, scope='Conv6_8')
            u = depthwise_conv(u, kernel=3, stride=2, padding='SAME',scope='Conv6_9')
            u = slim.conv2d(u, 64, (1, 1), stride=1, scope='Conv6_10')
            '''
            x = block(x, num_units=4, scope='Stage4')
            '''
            y = slim.conv2d(x, 64, (1, 1), stride=1, scope='Conv6_1')
            y = tf.concat([z,y],axis=3)
            y = depthwise_conv(y, kernel=3, stride=1, padding='VALID',scope='Conv6_2') #4
            y = slim.conv2d(y, 64, (1, 1), stride=1, scope='Conv6_3')
            y = depthwise_conv(y, kernel=3, stride=2, padding='SAME',scope='Conv6_4') #2
            y = slim.conv2d_transpose(y,64, (3,3), stride=[2,2],padding="VALID")
            x = slim.conv2d(x, 64, (1, 1), stride=1, scope='sConv')
            x = tf.concat([x,y,u],axis=3)
            '''
            #shape = tf.shape(x)
            #ch = shape[3]
            if False:
                with tf.variable_scope('RFBModule'):
                    x = RFBModuleB2(x, 192)

            if depth_multiplier == '0.33':
                final_channels = 512
            elif depth_multiplier == '2.0':
                final_channels = 2048
            else:
                final_channels = 1024
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')
    
    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])
    

    logits = slim.fully_connected(
        x, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits

def RFBModuleB(x, in_channels):
    inc=in_channels//8
    with tf.variable_scope('branch0'):
        conv1x1=slim.conv2d(x, 2*inc, (1, 1), stride=1, scope='conv1x1')
        branch0_conv3x3=slim.conv2d(conv1x1, 2*inc, (3, 3), stride=1, padding='SAME', activation_fn=None)
    with tf.variable_scope('branch1'):
        conv1x1=slim.conv2d(x, 1*inc, (1, 1), stride=1, scope='conv1x1')
        branch1_conv3x3=slim.conv2d(conv1x1, 2*inc, (3, 3), stride=1, padding='SAME')
        branch1_conv3x3_dilation=slim.conv2d(branch1_conv3x3, 2*inc, (3, 3), stride=1, padding='SAME',rate=2, activation_fn=None)
    with tf.variable_scope('branch2'):
        conv1x1=slim.conv2d(x, 1*inc, (1, 1), stride=1, scope='conv1x1')
        branch2_conv5x5_1=slim.conv2d(conv1x1, (inc//2)*3, (3, 3), stride=1, padding='SAME') 
        branch2_conv5x5_2=slim.conv2d(branch2_conv5x5_1, 2*inc, (3, 3), stride=1, padding='SAME') 
        branch2_conv3x3_dilation=slim.conv2d(branch2_conv5x5_2, 2*inc, (3, 3), stride=1, padding='SAME',rate=5,activation_fn=None)  
    shortcut=slim.conv2d(x, in_channels, (1, 1), stride=1, scope='shortcut',activation_fn=None)
    
    shape = tf.shape(shortcut)
    batch_size = shape[0]
    height, width = shape[1], shape[2]
    #depth = conv1x1.shape[3].value
    #[batch,height,width,4,depth]
    x = tf.stack([branch0_conv3x3,branch1_conv3x3_dilation,branch2_conv3x3_dilation], axis=3)
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [batch_size, height, width, 6*inc])
    x=slim.conv2d(x, in_channels, (1, 1), stride=1, scope='output',activation_fn=None)
    scale=tf.fill([batch_size,1,1,in_channels],1.0)
    x=x*scale+shortcut
    x=tf.nn.relu(x)
    return x

def RFBModuleB2(x, in_channels):
    inc=in_channels//8
    x, y, z, w = tf.split(x, num_or_size_splits=4, axis=3)
    with tf.variable_scope('branch0'):
        #conv1x1=slim.conv2d(x, 2*inc, (1, 1), stride=1, scope='conv1x1')
        branch0_conv3x3=slim.conv2d(y, 2*inc, (3, 3), stride=1, padding='SAME', activation_fn=None)
    with tf.variable_scope('branch1'):
        #conv1x1=slim.conv2d(x, 2*inc, (1, 1), stride=1, scope='conv1x1')
        branch1_conv3x3=depthwise_conv(z, kernel=3, stride=1, padding='SAME',activation_fn=tf.nn.relu)
        branch1_conv3x3_dilation=slim.conv2d(branch1_conv3x3, 2*inc, (3, 3), stride=1, padding='SAME',rate=2, activation_fn=None)
    with tf.variable_scope('branch2'):
        #conv1x1=slim.conv2d(x, 2*inc, (1, 1), stride=1, scope='conv1x1')
        branch2_conv5x5_1=depthwise_conv(w, kernel=3, stride=1, padding='SAME',activation_fn=tf.nn.relu) 
        branch2_conv5x5_2=depthwise_conv(branch2_conv5x5_1, kernel=3, stride=1, padding='SAME',activation_fn=tf.nn.relu,scope='depthwise_conv2')
        branch2_conv3x3_dilation=slim.conv2d(branch2_conv5x5_2, 2*inc, (3, 3), stride=1, padding='SAME',rate=5,activation_fn=None)  
    shortcut=slim.conv2d(x, 2*inc, (1, 1), stride=1, scope='shortcut',activation_fn=None)
    
    x = tf.concat([shortcut,branch0_conv3x3,branch1_conv3x3_dilation,branch2_conv3x3_dilation], axis=3)
    x=slim.conv2d(x, in_channels, (1, 1), stride=1, scope='output')
    return x

def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
        
    if False:  #with SENet module
        SEch=in_channels
        with tf.variable_scope('SEModule'):
            z= tf.reduce_mean(x, axis=[1, 2], name='globalPooling')
            z=slim.fully_connected(
                z, SEch // 2, activation_fn=tf.nn.relu, scope='fc1',
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            z=slim.fully_connected(
                z, SEch, activation_fn=tf.nn.sigmoid, scope='fc2',
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            z=tf.reshape(z,[-1,1,1,SEch])
            x=x*z
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
    
    SEch = out_channels //2
    if False:  #with SENet module
        with tf.variable_scope('SEModule'):
            z= tf.reduce_mean(y, axis=[1, 2], name='globalPooling')
            z=slim.fully_connected(
                z, SEch // 16, activation_fn=tf.nn.relu, scope='fc1',
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            z=slim.fully_connected(
                z, SEch, activation_fn=tf.nn.sigmoid, scope='fc2',
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            z=tf.reshape(z,[-1,1,1,out_channels // 2])
            y=y*z

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
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

