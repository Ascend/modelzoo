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

""" GradualPruneHook to prune weights gradually during training.
    gradual prune and layer_constant: https://arxiv.org/pdf/1710.01878.pdf
    blind and class-based prune: https://nlp.stanford.edu/pubs/see2016compression.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

from noahnmt.hooks.train_hooks import TrainingHook
from noahnmt.utils import constant_utils


class GradualPruneHook(TrainingHook):
  """Occasionally samples predictions from the training run and prints them.

  Params:
    every_n_secs: Sample predictions every N seconds.
      If set, `every_n_steps` must be None.
    every_n_steps: Sample predictions every N steps.
      If set, `every_n_secs` must be None.
    sample_dir: Optional, a directory to write samples to.
    delimiter: Join tokens on this delimiter. Defaults to space.
  """

  #pylint: disable=missing-docstring

  def __init__(self, params, model_dir, run_config):
    super(GradualPruneHook, self).__init__(params, model_dir, run_config)

    self._timer = SecondOrStepTimer(
        every_secs=None,
        every_steps=self.params["prune_every_n_steps"])

    self._should_trigger = False
    self._iter_count = 0
    self._global_step = None

    self._exclude_names = self.params["exclude_names"]
    self._final_prune_rate = self.params["final_prune_rate"]
    self._start_prune_step = self.params["start_prune_step"]
    self._stop_prune_step = self.params["stop_prune_step"]
    self._prune_method = self.params["prune_method"]
    self._layer_constant = self.params["layer_constant"]

  @staticmethod
  def default_params():
    return {
        "exclude_names": None,
        "final_prune_rate": 0.8,
        "prune_every_n_steps": 500,
        "start_prune_step": 0,
        "stop_prune_step": tf.int32.max,
        "prune_method": "uniform", # uniform or blind
        "layer_constant": False, # TODO implement this functionality
    }
  
  def _exclude(self, name):
    if self._exclude_names is not None:
      for n in self._exclude_names:
        if name.find(n) >= 0:
          return True
    return False

  def _get_threshold_tensor(self, tensor, rate):
    assert len(tensor.get_shape().as_list()) == 1
    length = tf.shape(tensor)[0]
    pos = tf.cast(
        tf.floor((1-rate) * tf.cast(length, dtype=constant_utils.DT_FLOAT())),
        dtype=constant_utils.DT_INT())
    topk_values, _ = tf.nn.top_k(tf.abs(tensor), k=length)
    threshold_value = topk_values[pos]
    return threshold_value


  def _get_threshold(self, weights, rate):
    thresholds = []
  
    if self._prune_method == "blind":
      flat_tensor = []
      with tf.device(self._global_step.device):
        x = tf.constant([0], dtype=constant_utils.DT_FLOAT())
        for var in weights:
          x = tf.concat([x, tf.reshape(var, [-1])], axis=0)
        
        threshold_value = self._get_threshold_tensor(x, rate)
        thresholds = [threshold_value] * len(weights)

    elif self._prune_method == "uniform":
      for i, var in enumerate(weights):
        with tf.device(var.device), tf.name_scope("threshold_%d" % i):
          x = tf.reshape(var, [-1])
          threshold_value = self._get_threshold_tensor(x, rate)
          thresholds.append(threshold_value)

    else:
      raise ValueError("Unknow prune method: blind or uniform.")

    return thresholds


  def _create_prune_op(self):
    global_step = tf.cast(
        self._global_step, dtype=constant_utils.DT_FLOAT())
    num_steps = float(self._stop_prune_step - self._start_prune_step)
    # current pruning rate
    current_rate = self._final_prune_rate * (
        1. - (1. - (global_step-float(self._start_prune_step)) / num_steps) ** 3)
    current_rate = tf.maximum(current_rate, 0.0)
    self._rate = current_rate
    # weights to be pruned
    weights = [var for var in tf.trainable_variables() if not self._exclude(var.name)]
    # get threshold to prune for each weight
    thresholds = self._get_threshold(weights, current_rate)

    self._weights = weights
    self._thresholds = thresholds

    # create ops
    mask_ops = []
    for var, value in zip(weights, thresholds):
      with tf.device(var.device):
        mask = tf.cast(
            tf.greater_equal(tf.abs(var), value), 
            dtype=constant_utils.DT_FLOAT())
        mask_ops.append(tf.assign(var, var * mask).op)
    return tf.group(*mask_ops)
  

  def _create_logging_info(self):
    
    def _calc_nonzero_rate(tensor):
      mask = tf.cast(tf.abs(tensor)>0, dtype=constant_utils.DT_FLOAT())
      return tf.reduce_sum(mask) / tf.size(tensor, out_type=constant_utils.DT_FLOAT())

    self._ratios = [_calc_nonzero_rate(w) for w in self._weights]
    self._min_values = [tf.reduce_min(w) for w in self._weights]
    self._max_values = [tf.reduce_max(w) for w in self._weights]


  def begin(self):
    self._iter_count = 0
    if self._exclude_names:
      self._exclude_names = self._exclude_names.split(",")
    self._global_step = tf.train.get_global_step()

    with tf.name_scope("gradual_prune_op"):
      self._prune_op = self._create_prune_op()
    self._create_logging_info()

  
  def after_create_session(self, session, coord):
    self._iter_count = session.run(self._global_step)


  def before_run(self, _run_context):
    if self._iter_count < self._start_prune_step \
        or self._iter_count > self._stop_prune_step:
      self._should_trigger = False
    else:
      self._should_trigger = self._timer.should_trigger_for_step(
          self._iter_count)             

    return tf.train.SessionRunArgs(self._global_step)


  def after_run(self, run_context, run_values):
    step = run_values.results
    self._iter_count = step

    if not self._should_trigger:
      return None

    # prune weights
    _, rate = run_context.session.run([self._prune_op, self._rate])
    tf.logging.info("Pruned model to rate: %f at step %d" % (rate, step))

    # logging info
    per_ratios, per_maxs, per_mins, per_thresholds = run_context.session.run(
        [self._ratios, self._max_values, self._min_values, self._thresholds])
    names = [w.name for w in self._weights]
    
    for name, ratio, maxv, minv, th in zip(names, per_ratios, per_maxs, per_mins, per_thresholds):
      tf.logging.info("non-zero = %.3f, max = %.3f, min = %.3f, threshold = %.3f, name = %s" % (ratio, maxv, minv, th, name))
    
    self._timer.update_last_triggered_step(self._iter_count - 1)