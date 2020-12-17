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

""" Implementation of sum attention layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six
import math

import tensorflow as tf

from noahnmt.attentions import attention
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.layers import common_layers as common_utils
from noahnmt.utils.transformer_utils import layer_norm

@registry.register_attention
class SumAttention(attention.Attention):
  """
  Attention layer according to https://arxiv.org/abs/1409.0473.
  """

  def __init__(self, params, mode, name="sum_attention"):
    super(SumAttention, self).__init__(params, mode, name)
    # indicator for reusing variables
    self._built_vars = False
    
  
  @staticmethod
  def default_params():
    return {
        "num_units": 512,
        "norm": False,
        "context_norm": False,
        "num_heads": 1,
        "use_bias": True,
        "dropout_rate": 0.2,
        "query_layer": True,
        "key_layer": True,
        "value_layer": True,
    }
    

  def prepare_memory(self, memory, memory_length=None):
    """ Loop invariant nodes
    Called before cls()
    """
    # require built
    if not self._built_vars:
      self._build_vars()

    self.memory = memory
    self.memory_length = memory_length
    self.keys = self.memory
    self.values = self.memory

    if memory_length is not None:
      self.scores_mask = tf.sequence_mask(
          lengths=tf.to_int32(self.memory_length),
          maxlen=tf.shape(self.memory)[1],
          dtype=constant_utils.DT_FLOAT()
      )
      self.scores_mask = (1. - self.scores_mask) * -constant_utils.INF
    else:
      self.scores_mask = None

    if self.params["key_layer"]:
      self.keys = self.key_layer(self.memory)

    if self.params["value_layer"]:
      self.values = self.value_layer(self.memory)
    
    self.max_time = tf.shape(self.values)[1]
    self.output_size = self.values.get_shape()[-1]
    
    # if multi-heads
    if self.params["num_heads"] > 1:
      if self.scores_mask is not None:
        self.scores_mask = tf.expand_dims(self.scores_mask, -1)
      # batch x length x num_heads x num_units/num_heads
      self.keys = common_utils.split_last_dim(
          self.keys, self.params["num_heads"])
      self.values = common_utils.split_last_dim(
          self.values, self.params["num_heads"])
      # # batch x num_heads x length x head_units
      # self.values = tf.transpose(self.values, [0,2,1,3])

  def _build_vars(self):
    self._built_vars = True
    # some projection layers
    if self.params["key_layer"]:
      self.key_layer = tf.layers.Dense(
          units=self.params["num_units"],
          use_bias=self.params["use_bias"],
          name="key_layer"
      )

    if self.params["value_layer"]:
      self.value_layer = tf.layers.Dense(
          units=self.params["num_units"],
          use_bias=self.params["use_bias"],
          name="value_layer"
      )
    
    if self.params["query_layer"]:
      self.query_layer = tf.layers.Dense(
          units=self.params["num_units"], 
          use_bias=self.params["use_bias"],
          name="query_layer",
      )

    # sum attention variables
    num_units = self.params["num_units"]
    num_heads = self.params["num_heads"]
    
    units_per_head = num_units // num_heads if num_heads > 0 else num_units

    self.q_att = tf.layers.Dense(
        units=units_per_head, 
        use_bias=False,
        name="query_att",
    )
    self.k_att = tf.layers.Dense(
        units=units_per_head, 
        use_bias=False,
        name="key_att",
    )
    self.v_att = tf.get_variable(
        name="v_att",
        shape=[units_per_head],
        dtype=constant_utils.DT_FLOAT()
    )

    if self.params["norm"]:
      # Scalar used in weight normalization
      self.g = tf.get_variable(
          name="attention_g", 
          dtype=constant_utils.DT_FLOAT(), 
          shape=[],
          initializer=tf.constant_initializer(
            value=math.sqrt((1. / units_per_head))))
      # Bias added prior to the nonlinearity
      self.b = tf.get_variable(
          name="attention_b", 
          shape=[units_per_head], 
          dtype=constant_utils.DT_FLOAT(),
          initializer=tf.zeros_initializer())


    self.fc = tf.layers.Dense(
          units=self.params["num_units"],
          use_bias=self.params["use_bias"],
          name="fc"
      )


  def score_fn(self, keys, query):
    keys = self.k_att(keys)
    query = self.q_att(query)
    
    v_att = self.v_att
    if self.params["norm"]:
      g = self.g
      b = self.b
      # normed_v = g * v / ||v||
      normed_v = g * v_att * tf.rsqrt(
          tf.reduce_sum(tf.square(v_att)))
      return tf.reduce_sum(
          normed_v * tf.tanh(keys + tf.expand_dims(query, 1) + b),
          -1)
    else:
      return tf.reduce_sum(
          v_att * tf.tanh(keys + tf.expand_dims(query, 1)),
          -1)


  def attend(self, query, memory=None, memory_length=None, **kwargs):
    """Computes attention scores and outputs.

    Args:
      query: The query used to calculate attention scores.
        In seq2seq this is typically the current state of the decoder.
        A tensor of shape `[B, ...]`
    Returns:
      A tuple `(scores, context)`.
      `scores` is vector of length `T` where each element is the
      normalized "score" of the corresponding `inputs` element.
      `context` is the final attention layer output corresponding to
      the weighted inputs.
      A tensor fo shape `[B, input_dim]`.
    """
    # create variables
    if not self._built_vars:
      self._build_vars()

    if memory is not None:
      self.prepare_memory(memory=memory, memory_length=memory_length)

    if self.params["query_layer"]:
      query = self.query_layer(query)

    if self.params["num_heads"] > 1:
      # batch x num_heads x num_units//num_heads
      query = common_utils.split_last_dim(query, self.params["num_heads"])

    # batch x length [x num_heads]
    scores = self.score_fn(self.keys, query)

    # Mask and normalize the scores
    if self.scores_mask is not None:
      scores = scores + self.scores_mask
    scores_normalized = tf.nn.softmax(
        scores, 
        name="scores_normalized",
        axis=1)

    if self.mode == tf.estimator.ModeKeys.TRAIN and self.params["dropout_rate"] > 0.:
      scores_normalized = tf.layers.dropout(
          inputs=scores_normalized, 
          rate=self.params["dropout_rate"],
          training=True)

    # Calculate the weighted average of the attention inputs
    # according to the scores
    if self.params["num_heads"] > 1:
      # batch x length x num_heads x head_units
      context = tf.expand_dims(scores_normalized, -1) * self.values
      context = tf.reduce_sum(context, 1, name="context")
      # batch x num_heads x head_units --> batch x units
      context = common_utils.combine_last_two_dims(context)
      scores_normalized = tf.reduce_mean(scores_normalized, axis=2)
    else:
      context = tf.expand_dims(scores_normalized, 2) * self.values
      context = tf.reduce_sum(context, 1, name="context")

    context = self.fc(context)

    if self.params["context_norm"]:
        context = layer_norm(context)




    context.set_shape([None, self.output_size])
    scores_normalized.set_shape([None, None])

    return scores_normalized, context
