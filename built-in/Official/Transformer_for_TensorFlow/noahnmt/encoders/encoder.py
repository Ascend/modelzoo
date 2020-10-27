# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

"""
Abstract base class for encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import six

from noahnmt.configurable import Configurable
from noahnmt.graph_module import GraphModule


ENCODER_OUTPUT = "encoder_output"
FINAL_STATE = "final_state"
ENCODER_MEM = "encoder_mem"


@six.add_metaclass(abc.ABCMeta)
class Encoder(GraphModule, Configurable):
  """Abstract encoder class. All encoders should inherit from this.

  Args:
    params: A dictionary of hyperparameters for the encoder.
    name: A variable scope for the encoder graph.
  """

  def __init__(self, params, mode, name):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)

  def _build(self, inputs, *args, **kwargs):
    return self.encode(inputs, *args, **kwargs)

  @abc.abstractmethod
  def encode(self, inputs, sequence_length, **kwargs):
    """
    Encodes an input sequence.

    Args:
      inputs: The inputs to encode. A float32 tensor of shape [B, T, ...].
      sequence_length: The length of each input. An int32 tensor of shape [T].

    Returns:
      An `EncoderOutput` tuple containing the outputs and final state.
    """
    raise NotImplementedError
