#!/usr/bin/env python
# coding=utf-8
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from thumt.layers.nn import linear

 
class GPULegacyGRUCell(tf.nn.rnn_cell.RNNCell):
    """ Groundhog's implementation of GRUCell
    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    """

    def __init__(self, num_units, reuse=None, input_type=None):
        super(GPULegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="reset_gate"))
            u = tf.nn.sigmoid(linear(all_inputs, self._num_units, False, False,
                                     scope="update_gate"))
            all_inputs = list(inputs) + [r * state]
            c = linear(all_inputs, self._num_units, True, False,
                       scope="candidate")

            new_state = (1.0 - u) * state + u * tf.tanh(c)

        return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units



class LegacyGRUCell(tf.nn.rnn_cell.RNNCell):
    """ NPU version of Groundhog's implementation of GRUCell
    :param num_units: int, The number of units in the RNN cell.
    :param reuse: (optional) Python boolean describing whether to reuse
        variables in an existing scope.  If not `True`, and the existing
        scope already has the given variables, an error is raised.
    :param input_type: default is 2-type: hidden state and state for encoder,
        but usually input_type=3 for the decoder
    """

    def __init__(self, num_units, reuse=None, input_type=2):
        super(LegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.reuse = reuse

        self.linear1 = [layers_core.Dense(self._num_units, use_bias=False, name="reset_gate_%d" %i, trainable=True
            ) for i in range(input_type)]
        self.linear2 = [layers_core.Dense(self._num_units, use_bias=False, name="update_gate_%d" %i, trainable=True
            ) for i in range(input_type)]
        self.linear3 = [layers_core.Dense(self._num_units, use_bias=False, name="candidate_%d" %i, trainable=True
            ) for i in range(input_type)]

        self.bias = self.add_weight(    
          'candidate/bias',
          shape=[self._num_units,],
        )
            

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="gru_cell",
                               values=[inputs, state]):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            all_inputs = list(inputs) + [state]
            r = tf.nn.sigmoid(self.dense(self.linear1, all_inputs, self._num_units, False))
            u = tf.nn.sigmoid(self.dense(self.linear2, all_inputs, self._num_units, False))
            all_inputs = list(inputs) + [r * state]
            c = self.dense(self.linear3, all_inputs, self._num_units, True)

            new_state = (1.0 - u) * state + u * tf.tanh(c)

        return new_state, new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def dense(self, linear_module, inputs, output_size, bias, dtype=None, scope=None):
        """
        Linear layer
        :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
        :param output_size: An integer specify the output size
        :param bias: a boolean value indicate whether to use bias term
        :param concat: a boolean value indicate whether to concatenate all inputs
        :param dtype: an instance of tf.DType
        :param scope: the scope of this layer, the default value is ``linear''
        :returns: a Tensor with shape [batch, output_size]
        :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                              compatible with each other
        """
        with tf.variable_scope(scope, default_name="linear", values=[inputs], dtype=dtype):#, reuse=self.reuse):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            input_size = [item.get_shape()[-1].value for item in inputs]
            
            if len(inputs) != len(input_size):
                raise RuntimeError("inputs and input_size unmatched!")

            output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]], axis=0)
            # Flatten to 2D
            inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]
            
            results = []
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                results.append(linear_module[i](inputs[i]))
           
            output = tf.add_n(results)
            if len(input_size) == 3:
                output = tf.nn.bias_add(output, self.bias)
            output = tf.reshape(output, output_shape)
            return output



