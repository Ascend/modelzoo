# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
from npu_bridge.estimator import npu_ops 

def inference_alexnet_impl(inputs, num_classes=1000, is_training=True):

    x = inputs
    # conv11*11
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4), padding='valid',use_bias=True,activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')

    # conv5*5
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')

    # conv3*3
    x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu)

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu)

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu)

    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')

    x = tf.reshape(x, [-1, 256*6*6])

    # fc layers
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True)
    if is_training:
        x = npu_ops.dropout(x, 0.65)
    else:
        x = npu_ops.dropout(x, 1.0)
    
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True)
    if is_training:
        x = npu_ops.dropout(x, 0.65)
    else:
        x = npu_ops.dropout(x, 1.0)
    
    x = tf.layers.dense(x, num_classes, activation=tf.nn.relu, use_bias=True)

    return x

def inference_alexnet_impl_he_uniform(inputs,num_classes=1000, is_training=True):

    x = inputs
    # conv11*11
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4), padding='valid',use_bias=True,activation=tf.nn.relu, kernel_initializer=tf.initializers.he_uniform(5))
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')

    # conv5*5
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))
    x = tf.layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')

    # conv3*3
    x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,kernel_initializer=tf.initializers.he_uniform(5))

    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')

    x = tf.reshape(x, [-1, 256*6*6])

    # fc layers
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))

    if is_training:
        x = npu_ops.dropout(x, 0.65)
    else:
        x = npu_ops.dropout(x, 1.0)

    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))
    
    if is_training:
        x = npu_ops.dropout(x, 0.65)
    else:
        x = npu_ops.dropout(x, 1.0)

    x = tf.layers.dense(x, num_classes, activation=tf.nn.relu, use_bias=True,kernel_initializer=tf.initializers.he_uniform(5))

    return x

def inference_alexnet_impl_he_uniform_custom(inputs,num_classes=1000, is_training=True):
    '''
      to be consistent with ME  default weight initialization

    '''

    scale =1.0/3.0 
    x = inputs
    # conv11*11
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=(4,4),
                         padding='valid',use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    x = tf.layers.max_pooling2d(x,pool_size=(3,3),strides=(2,2), padding='valid')

    # conv5*5
    x = tf.pad(x, paddings=[[0,0],[2,2],[2,2],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=192, kernel_size=5, strides=(1,1), padding='valid',use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    x = tf.layers.max_pooling2d(x, pool_size=(3 ,3), strides=(2,2),  padding='valid')

    # conv3*3
    x = tf.pad(x, paddings=[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=384, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=(1, 1), padding='valid', use_bias=True,activation=tf.nn.relu,
                         kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding='valid')

    x = tf.reshape(x, [-1, 256*6*6])

    # fc layers
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    if is_training:
        x = npu_ops.dropout(x, 0.65)
    else:
        x = npu_ops.dropout(x, 1.0)
    
    x = tf.layers.dense(x, 4096, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))
    if is_training:
        x = npu_ops.dropout(x, 0.65)
    else:
        x = npu_ops.dropout(x, 1.0)

    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer= tf.variance_scaling_initializer(scale=scale, mode ='fan_in',distribution='uniform'))

    return x
    
def inference(inputs,version="xavier",num_classes=1000, is_training=False):


    if version=='xavier':
        return  inference_alexnet_impl(inputs,num_classes, is_training=is_training)
    elif version=='he_uniform':
        return  inference_alexnet_impl_he_uniform(inputs,num_classes, is_training=is_training)
    elif version=='he_uniform_custom':
        return inference_alexnet_impl_he_uniform_custom(inputs, num_classes,is_training=is_training)
    else:
        raise ValueError('Invalid type of version , should be one of following: xavier, he_uniform, he_uniform_custom')

