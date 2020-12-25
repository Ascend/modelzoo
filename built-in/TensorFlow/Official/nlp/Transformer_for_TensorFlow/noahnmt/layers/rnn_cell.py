# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from noahnmt.layers import common_layers

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class LGRUCell(tf.nn.rnn_cell.GRUCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               layer_norm=False,
               dropout_rate=None):
    super(LGRUCell, self).__init__(
        num_units=num_units,
        activation=activation,
        reuse=reuse,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name)
    self._layer_norm = layer_norm
    self._ln_epsilon = 1e-6
    self._dropout_rate = dropout_rate


  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 3 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[3 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else tf.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else tf.zeros_initializer(dtype=self.dtype)))

    self._linear_kernel = self.add_variable(
        "linear/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth, self._num_units],
        initializer=self._kernel_initializer)

    if self._layer_norm:
      self._ln_scale = self.add_variable(
          "layer_norm/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[3 * self._num_units],
          initializer=tf.ones_initializer(dtype=self.dtype))
      self._ln_bias = self.add_variable(
          "layer_norm/%s" % _BIAS_VARIABLE_NAME,
          shape=[3 * self._num_units],
          initializer=tf.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    gate_inputs = tf.matmul(
        tf.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

    if self._layer_norm:
      gate_inputs = common_layers.split_last_dim(gate_inputs, 3)
      mean = tf.reduce_mean(gate_inputs, axis=[-1], keepdims=True)
      variance = tf.reduce_mean(tf.square(gate_inputs - mean), axis=[-1], keepdims=True)
      norm_x = (gate_inputs - mean) * tf.rsqrt(variance + self._ln_epsilon)
      norm_x = common_layers.combine_last_two_dims(norm_x)
      gate_inputs = norm_x * self._ln_scale + self._ln_bias

    value = tf.sigmoid(gate_inputs)
    r, u, l = tf.split(value=value, num_or_size_splits=3, axis=1)

    r_state = r * state

    candidate = tf.matmul(
        tf.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = tf.nn.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    c += l * tf.matmul(inputs, self._linear_kernel)
    if self._dropout_rate:
      c = tf.nn.dropout(c, keep_prob=1-self._dropout_rate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class TGRUCell(tf.nn.rnn_cell.GRUCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               layer_norm=False,
               dropout_rate=None):
    super(TGRUCell, self).__init__(
        num_units=num_units,
        activation=activation,
        reuse=reuse,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name)
    self._layer_norm = layer_norm
    self._ln_epsilon = 1e-6
    self._dropout_rate = dropout_rate


  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    # input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else tf.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else tf.zeros_initializer(dtype=self.dtype)))

    if self._layer_norm:
      self._ln_scale = self.add_variable(
          "layer_norm/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[2 * self._num_units],
          initializer=tf.ones_initializer(dtype=self.dtype))
      self._ln_bias = self.add_variable(
          "layer_norm/%s" % _BIAS_VARIABLE_NAME,
          shape=[2 * self._num_units],
          initializer=tf.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""

    gate_inputs = tf.matmul(state, self._gate_kernel)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

    if self._layer_norm:
      gate_inputs = common_layers.split_last_dim(gate_inputs, 2)
      mean = tf.reduce_mean(gate_inputs, axis=[-1], keepdims=True)
      variance = tf.reduce_mean(tf.square(gate_inputs - mean), axis=[-1], keepdims=True)
      norm_x = (gate_inputs - mean) * tf.rsqrt(variance + self._ln_epsilon)
      norm_x = common_layers.combine_last_two_dims(norm_x)
      gate_inputs = norm_x * self._ln_scale + self._ln_bias

    value = tf.sigmoid(gate_inputs)
    r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = tf.matmul(r_state, self._candidate_kernel)
    candidate = tf.nn.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    if self._dropout_rate:
      c = tf.nn.dropout(c, keep_prob=1-self._dropout_rate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class TransitionRNNCell(tf.nn.rnn_cell.MultiRNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=False):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(TransitionRNNCell, self).__init__(cells, state_is_tuple)


  @property
  def state_size(self):
    return self._cells[-1].state_size
  
  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cells[-1].zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_inp = inputs
    cur_state = state

    for i, cell in enumerate(self._cells):
      with tf.variable_scope("cell_%d" % i):
        cur_inp, cur_state = cell(cur_inp, cur_state)

    return cur_inp, cur_state