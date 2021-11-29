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
