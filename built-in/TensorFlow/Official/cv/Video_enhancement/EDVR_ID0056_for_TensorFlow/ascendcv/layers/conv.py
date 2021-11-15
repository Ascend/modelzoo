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
import math

import tensorflow as tf

from ..utils.initializer import get_initializer, calculate_fan


def Conv2D(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilations=(1, 1), use_bias=True,
           kernel_initializer=None, bias_initializer=None,
           trainable=True, name='Conv2D'):

    if kernel_initializer is None:
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), int(x.shape[-1]), filters, kernel_size)
    if bias_initializer is None:
        fan = calculate_fan(kernel_size, int(x.shape[-1]))
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(-bound, bound)

    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        dilation_rate=dilations,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        name=name,
    )
    return x


def Conv3D(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', dilations=(1, 1, 1), use_bias=True,
           kernel_initializer=None, bias_initializer=None,
           trainable=True, name='Conv2D'):

    if kernel_initializer is None:
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), int(x.shape[-1]), filters, kernel_size)
    if bias_initializer is None:
        fan = calculate_fan(kernel_size, int(x.shape[-1]))
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(-bound, bound)

    x = tf.layers.conv3d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        dilation_rate=dilations,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        name=name,
    )
    return x
