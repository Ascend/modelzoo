# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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
#
"""Miscellaneous training utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import defaultdict
import json

import tensorflow as tf
from tensorflow import gfile


class TrainOptions(object):
  """A collection of options that are passed to the training script
  and can be saved to perform inference later.

  Args:
    task: Name of the training task class.
    task_params: A dictionary of parameters passed to the training task.
  """

  def __init__(self, model_name, model_params):
    self._model_name = model_name
    self._model_params = model_params

  @property
  def model_name(self):
    """Returns the training task parameters"""
    return self._model_name

  @property
  def model_params(self):
    """Returns the training task class"""
    return self._model_params

  @staticmethod
  def path(model_dir):
    """Returns the path to the options file.

    Args:
      model_dir: The model directory
    """
    return os.path.join(model_dir, "train_options.json")

  def dump(self, model_dir):
    """Dumps the options to a file in the model directory.

    Args:
      model_dir: Path to the model directory. The options will be
      dumped into a file in this directory.
    """
    gfile.MakeDirs(model_dir)
    options_dict = {
        "model_name": self.model_name,
        "model_params": self.model_params,
    }

    with gfile.GFile(TrainOptions.path(model_dir), "wb") as file:
      file.write(json.dumps(options_dict).encode("utf-8"))

  @staticmethod
  def load(model_dir):
    """ Loads options from the given model directory.

    Args:
      model_dir: Path to the model directory.
    """
    with gfile.GFile(TrainOptions.path(model_dir), "rb") as file:
      options_dict = json.loads(file.read().decode("utf-8"))
    options_dict = defaultdict(None, options_dict)

    return TrainOptions(
        model_name=options_dict["model_name"],
        model_params=options_dict["model_params"])
