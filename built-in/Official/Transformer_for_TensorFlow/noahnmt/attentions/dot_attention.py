# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

""" Implementation of dot attention layer.
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
class DotAttention(SumAttention):
  """ (Scaled) dot attention as in loung's paper
  """

  def __init__(self, params, mode, name="dot_attention"):
    super(DotAttention, self).__init__(
        params, mode, NameError)
  

  @staticmethod
  def default_params():
    params = SumAttention.default_params()
    params.update({
        "query_layer": False,
        "key_layer": False,
        "value_layer": False,
    })
    return params


  def _build_vars(self):
    self._built_vars = True
    # variables
    if self.params["norm"]:
      # Scalar used in weight normalization
      self.g = tf.get_variable(
        "attention_g", dtype=constant_utils.DT_FLOAT(), shape=[],
        initializer=tf.constant_initializer(
            value=1.))


  def score_fn(self, keys, query):
    score = tf.reduce_sum(keys * tf.expand_dims(query, 1), -1)
    # Scalar used in weight scaling
    if self.params["norm"]:
      score = self.g * score
    return score
