# coding=utf-8
# Copyright Huawei Noah's Ark Lab.

"""
Abstract base class for inference tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc

import six
import tensorflow as tf

from noahnmt.configurable import Configurable, abstractstaticmethod
from noahnmt.utils import graph_utils


def unbatch_dict(dict_):
  """Converts a dictionary of batch items to a batch/list of
  dictionary items.
  """
  batch_size = list(dict_.values())[0].shape[0]
  for i in range(batch_size):
    yield {key: value[i] for key, value in dict_.items()}


@six.add_metaclass(abc.ABCMeta)
class InferenceTask(Configurable):
  """
  Abstract base class for inference tasks. Defines the logic used to make
  predictions for a specific type of task.

  Params:
    model_class: The model class to instantiate. If undefined,
      re-uses the class used during training.
    model_params: Model hyperparameters. Specified hyperparameters will
      overwrite those used during training.

  Args:
    params: See Params above.
  """

  def __init__(self, params):
    Configurable.__init__(self, params, tf.estimator.ModeKeys.PREDICT)


  def __call__(self, predictions):
    self.execute(predictions)
  

  @abc.abstractmethod
  def execute(self, predictions):
    raise NotImplementedError()
  

  def finalize(self):
    pass


  @abstractstaticmethod
  def default_params():
    raise NotImplementedError()
