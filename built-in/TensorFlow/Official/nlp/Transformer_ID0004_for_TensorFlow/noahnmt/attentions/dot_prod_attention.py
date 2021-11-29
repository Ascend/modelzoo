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

""" Implementation of dot-prod attention layer as in Transformer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six
import math

import tensorflow as tf


from noahnmt.attentions.sum_attention import SumAttention
from noahnmt.utils import constant_utils
from noahnmt.utils import registry
from noahnmt.layers import common_layers as common_utils


@registry.register_attention
class DotProdAttention(SumAttention):
  """ (Scaled) dot attention as in loung's paper
  """

  def __init__(self, params, mode, name="dot_prod_attention",
               memory=None, memory_length=None):
    super(DotProdAttention, self).__init__(
        params, mode, name)
  

  @staticmethod
  def default_params():
    params = SumAttention.default_params()
    params.update({
        "query_layer": True,
        "key_layer": True,
        "value_layer": True,
        "scaled": True,
        "num_heads": 8,
    })
    return params

  def prepare_memory(self, memory, memory_length=None):
    super(DotProdAttention, self).prepare_memory(
        memory, memory_length)
    if self.params["num_heads"] > 1:
      # batch x num_heads x length x units
      self.keys = tf.transpose(self.keys, [0, 2, 1, 3])


  def _build_vars(self):
    self._built_vars = True


  def score_fn(self, keys, query):
    if self.params["scaled"]:
      units_per_head = self.params["num_units"] // self.params["num_heads"]
      query *= units_per_head**-0.5
    
    # batch [x num_heads] x head_units x 1
    query = tf.expand_dims(query, axis=-1)
    # batch [x num_heads] x length
    score = tf.squeeze(tf.matmul(keys, query), axis=[-1])

    if self.params["num_heads"]> 1:
      # to batch x length x num_heads
      score = tf.transpose(score, [0, 2, 1])
      
    return score
