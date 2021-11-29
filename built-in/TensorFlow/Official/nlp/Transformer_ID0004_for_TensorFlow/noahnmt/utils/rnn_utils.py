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
from tensorflow.python.framework import ops
import tensorflow as tf
from noahnmt.layers import rnn_cell
from noahnmt.utils import transformer_utils as t2t_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes


class LayernormWrapper(tf.nn.rnn_cell.RNNCell):
  """Cell wrapper to add dropout within the output of LSTM cells."""

  def __init__(self, cell, name, _norm_gain=1.0, _norm_shift=0.0,  dtype=None):
    # Set cell, variational_recurrent, seed before running the code below
    self._cell = cell
    self.var_name = name
    self._norm_gain = _norm_gain
    self._norm_shift = _norm_shift

  @property
  def wrapped_cell(self):
    return self._cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)


  def _norm(self,inp,scope):
      """Run layer norm within the output of lstm cells"""
      shape = inp.get_shape()[-1:]
      dtype = inp.dtype
      gamma_init = init_ops.constant_initializer(self._norm_gain)
      beta_init = init_ops.constant_initializer(self._norm_shift)
      with vs.variable_scope(scope):
          # Initialize beta and gamma for use by layer_norm.
          vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
          vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
      normalized = layers.layer_norm(inp, reuse=True, scope=scope)
      return normalized

  def __call__(self, inputs, state, scope=None):
    """aplly layer normalisation within output of LSTM cells"""
    output, new_state = self._cell(inputs, state, scope=scope)
#     output = self._norm(output, self.var_name)
    output = t2t_utils.layer_norm(output, name=self.var_name)
    return output, new_state


class FeedforwardWrapper(tf.nn.rnn_cell.RNNCell):
    """Cell wrapper in include one feedforward layer after each lstm layer output.
    i.e. y = ffn2(relu(ffn1(x)))
    """

    def __init__(self, cell, n_units, ffn_units, use_bias=True, mode=None, dropout_rate=0.1, dtype=None,
                 activation="relu"):
        # Set cell, variational_recurrent, seed before running the code below
        self._cell = cell
        act_fn = None
        if activation == "relu":
            print("use relu!!!!!!!!!!!!!!!!!!!")
            act_fn = tf.nn.relu

        self.ffn1 = tf.layers.Dense(
            units=ffn_units,
            use_bias=use_bias,
            activation=act_fn,
            name="ffn1"
        )
        self.ffn2 = tf.layers.Dense(
            units=n_units,
            use_bias=use_bias,
            name="ffn2"
        )

        self.mode = mode
        self.dropout_rate = dropout_rate

    @property
    def wrapped_cell(self):
        return self._cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        """add feedforward layer similar to transformer arch; wraps dropout and residual"""
        assert self.dropout_rate > 0, "Dropout rate must be within range [0,1]"
        outputs, new_state = self._cell(inputs, state, scope=scope)
        residual = outputs
        outputs = self.ffn1(outputs)
        outputs = self.ffn2(outputs)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            outputs = tf.layers.dropout(
                inputs=outputs,
                rate=self.dropout_rate
            )
        outputs = outputs + residual
        return outputs, new_state

def default_rnn_cell_params():
  """Creates default parameters used by multiple RNN encoders.
  """
  return {
      "rnn.cell_type": "lstm",
      "layer_norm": False,
      "num_layers": 4,
      "num_units": 512,
      "dropout_rate": 0.2,
      "forget_bias": 1.0,
      "residual": True,
      "residual.start_layer": 2, # starting from 0
      "residual.custom_fn": None,
  }


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus <= 1:
    return None
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


def create_single_cell(
    mode,
    unit_type,
    num_units,
    forget_bias=1.0,
    dropout_rate=0,
    residual=False,
    layer_norm=False,
    ln_wrapper=False,
    ffn_wrapper=False,
    ffn_units=2048,
    ffn_act=None,
    residual_fn=None):
  """Create an instance of a single RNN cell."""

  # Cell Type
  if unit_type == "lstm":
    if layer_norm:
      single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
          num_units,
          forget_bias=forget_bias)
    else:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units,
          forget_bias=forget_bias)
  elif unit_type == "gru":
    single_cell = tf.nn.rnn_cell.GRUCell(num_units)
  elif unit_type == "lgru":
    single_cell = rnn_cell.LGRUCell(
        num_units,
        layer_norm=layer_norm,
        dropout_rate=dropout_rate)
    dropout_rate = None
  elif unit_type == "tgru":
    single_cell = rnn_cell.TGRUCell(
        num_units,
        layer_norm=layer_norm,
        dropout_rate=dropout_rate)
    dropout_rate = None
  elif unit_type == "sru":
    single_cell = tf.nn.rnn_cell.SRUCell(num_units)
  elif unit_type == "rnn":
    single_cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob)
  if mode == tf.estimator.ModeKeys.TRAIN and dropout_rate and dropout_rate > 0:
    single_cell = tf.nn.rnn_cell.DropoutWrapper(
    cell=single_cell, output_keep_prob=(1.0 - dropout_rate))

  # Residual
  if residual:
    single_cell = tf.nn.rnn_cell.ResidualWrapper(
        single_cell, residual_fn=residual_fn)
  # Layernorm(x+dp(attn(x)))
  if ln_wrapper:
    single_cell = LayernormWrapper(
            cell=single_cell, name="attn_ln"
        )

  # Add feedforward layer; wraps with dropout and residual
  if ffn_wrapper:
      single_cell = FeedforwardWrapper(
          cell=single_cell,
          mode=mode,
          dropout_rate=dropout_rate,
          ffn_units=ffn_units,
          n_units=num_units,
          activation=ffn_act
      )
  # # Layernorm(x+dp(ffn(x)))
  if ln_wrapper:
    single_cell = LayernormWrapper(
            cell=single_cell, name="ffn_ln"
        )



  return single_cell


def create_cell_list(
    mode,
    unit_type,
    num_units,
    num_layers,
    forget_bias=1.0,
    dropout_rate=0,
    layer_norm=False,
    ffn_wrapper=False,
    ffn_units=2048,
    ffn_act=None,
    ln_wrapper=False,
    residual=True,
    residual_start=2,
    residual_fn=None):
  """Create a list of RNN cells."""
  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    single_cell = create_single_cell(
        mode=mode,
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout_rate=dropout_rate,
        residual=(residual and i >= residual_start),
        layer_norm=layer_norm,
        ffn_wrapper=ffn_wrapper,
        ffn_units=ffn_units,
        ffn_act=ffn_act,
        ln_wrapper=ln_wrapper,
        residual_fn=residual_fn
    )
    # tf.logging.info("\n")
    cell_list.append(single_cell)

  return cell_list


def create_multi_rnn_cell(
    mode,
    unit_type,
    num_units,
    num_layers,
    forget_bias=1.0,
    dropout_rate=0,
    layer_norm=False,
    ffn_wrapper=False,
    ffn_units=2048,
    ffn_act=None,
    ln_wrapper=False,
    residual=True,
    residual_start=2,
    residual_fn=None):
  """Create multi-layer RNN cell.
  """

  cell_list = create_cell_list(
      mode=mode,
      unit_type=unit_type,
      num_units=num_units,
      num_layers=num_layers,
      forget_bias=forget_bias,
      dropout_rate=dropout_rate,
      layer_norm=layer_norm,
      ffn_wrapper=ffn_wrapper,
      ffn_units=ffn_units,
      ffn_act=ffn_act,
      ln_wrapper=ln_wrapper,
      residual=residual,
      residual_start=residual_start,
      residual_fn=residual_fn
  )

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  return tf.nn.rnn_cell.MultiRNNCell(cell_list)
