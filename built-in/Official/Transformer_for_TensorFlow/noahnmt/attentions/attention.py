# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

""" Base attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six

from noahnmt.graph_module import GraphModule
from noahnmt.configurable import Configurable

INF = 1. * 1e9

@six.add_metaclass(abc.ABCMeta)
class Attention(GraphModule, Configurable):
  """
  Attention layer according to https://arxiv.org/abs/1409.0473.

  Params:
    num_units: Number of units used in the attention layer
  """

  def __init__(self, params, mode, name, **kwargs):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)

  def _build(self, query, *args, **kwargs):
    return self.attend(query, *args, **kwargs)

  @abc.abstractmethod
  def attend(self, query, **kwargs):
    """
    return attention_scores and attention_context
    """
    raise NotImplementedError