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

import tensorflow as tf

from noahnmt.hooks.train_hooks import TrainingHook
from noahnmt.utils import graph_utils


class WpsCounterHook(tf.train.SessionRunHook):
  """Hook that counts steps per second."""

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps,
                                    every_secs=every_n_secs)

    # self._summary_writer = summary_writer
    # self._output_dir = output_dir
    self._words_tensor = None
    self._total_num_tokens = 0

  def begin(self):
    predictions = graph_utils.get_dict_from_collection("predictions")
    # for logging WPS
    self._words_tensor = tf.reduce_sum(predictions["target_mask"]) + tf.reduce_sum(predictions["source_mask"])

    self._global_step_tensor = tf.train.get_global_step()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")
    self._summary_tag = "words/sec"

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs([self._global_step_tensor, self._words_tensor])

  def after_run(self, run_context, run_values):
    _ = run_context

    stale_global_step, num_tokens = run_values.results
    self._total_num_tokens += num_tokens
    if self._timer.should_trigger_for_step(stale_global_step+1):
      # get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        elapsed_time, _ = self._timer.update_last_triggered_step(
            global_step)
        total_tokens = self._total_num_tokens
        self._total_num_tokens = 0
        if elapsed_time is not None:
          wps = total_tokens / elapsed_time / 1000
          tf.logging.info("%s: %.2fk", self._summary_tag, wps)
