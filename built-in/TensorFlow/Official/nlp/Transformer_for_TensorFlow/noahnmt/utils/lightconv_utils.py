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
from noahnmt.utils import constant_utils

def glu_fn(inputs, name="glu_fn"):
    """ Gated Linear Units
    """
    with tf.variable_scope(name):
        # split on the last dim
        input_pass, input_gate = tf.split(
            value=inputs,
            num_or_size_splits=2,
            axis=-1)
        # GLU(AB) = A * sigmoid(B)
        outputs = input_pass * tf.sigmoid(input_gate)
    return outputs


def linear_layer(inputs, units, glu=True, name="linear"):
    x = inputs
    # linear projection and optional GLU
    if glu:
        x = tf.layers.dense(
            inputs=x,
            units=2 * units,
            use_bias=True,
            name=name)
        x = glu_fn(x)
    else:
        x = tf.layers.dense(
            inputs=x,
            units=units,
            use_bias=True,
            name=name)
    return x


def light_conv_layer(inputs, kernel_size, num_heads, weight_softmax=True, name="light_conv", cache=None):
    batch, length, _, units = inputs.get_shape().as_list()
    units_per_head = units // num_heads

    with tf.variable_scope(name):
        weight = tf.get_variable(
            name="weight",
            shape=[num_heads, 1, kernel_size],
            dtype=inputs.dtype)

        if weight_softmax:
            weight = tf.nn.softmax(weight)

        # reshape and transpose weight to [h, w, ci, cm]
        weight = tf.reshape(
            tf.transpose(
                tf.tile(weight, [1, units_per_head, 1]),
                [2, 1, 0]),
            [kernel_size, 1, units, 1])

        # incremental state
        if cache is not None:
            inputs = tf.concat([inputs, cache["input"]], axis=1)
        
        is_half = (inputs.dtype == tf.float16)
        if is_half:
          weight = tf.cast(weight, tf.float32)
          inputs = tf.cast(inputs, tf.float32)
            
             
        output = tf.nn.depthwise_conv2d(
            input=inputs,
            filter=weight,
            strides=[1, 1, 1, 1],
            padding='SAME',
            rate=[1, 1],
            data_format='NHWC')
        
        if is_half:
          output = tf.cast(output, tf.float16)

        
        # update cache and slice output
        if cache is not None:
            left_len = kernel_size // 2
            _, clen, _, _ = cache["input"].get_shape().as_list()

            if clen >= left_len:
                cache["input"] = inputs[:, -left_len:]
            else:
                cache["input"] = inputs

            output = output[:, -1:]

        return output


def dynamic_conv_layer(inputs, kernel_size, num_heads, weight_softmax=True, name="dynamic_conv", cache=None):
    """
    inputs: shape [batch, length, 1, units]
    """
    pass