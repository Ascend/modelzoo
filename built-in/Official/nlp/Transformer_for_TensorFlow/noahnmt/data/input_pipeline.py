# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

"""
Collection of input pipelines.

An input pipeline defines how to read and parse data. It produces a tuple
of (features, labels) that can be read by tf.learn estimators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sys
import six
import math
import numpy as np

import tensorflow as tf

from noahnmt.configurable import Configurable
from noahnmt.utils import vocab_utils
# from noahnmt.utils import align_utils
from noahnmt.utils import constant_utils as const_utils
# from noahnmt.utils.sample_input_utils import create_sampling_dataset


@six.add_metaclass(abc.ABCMeta)
class InputPipeline(Configurable):
  """Abstract InputPipeline class. All input pipelines must inherit from this.
  An InputPipeline defines how data is read, parsed, and separated into
  features and labels.

  Params:
    shuffle: If true, shuffle the data.
    num_epochs: Number of times to iterate through the dataset. If None,
      iterate forever.
  """

  def __init__(self, params, mode):
    if "batch_multiplier" in params and params["batch_multiplier"] < 1:
      params["batch_multiplier"] = 1
    super(InputPipeline, self).__init__(params, mode)

    self.params["num_threads"] = self.params["num_threads"] * self.params["batch_multiplier"]

  @staticmethod
  def default_params():
    return {
        "num_threads": 4,
        "output_buffer_size": None,
        "batch_size": 32,
        "batch_multiplier": 1,
        "num_shards": 1,
        "shard_index": 0,
        "sos": vocab_utils.SOS,
        "eos": vocab_utils.EOS,
        "source_sos": False,
    }


  def read_data(self, **kwargs):
    """Creates DataProvider instance for this input pipeline. Additional
    keyword arguments are passed to the DataProvider.
    """
    raise NotImplementedError("Not implemented.")

