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
Copied from tensorflow with some modifications.
"""

import os
import copy
import numpy
import numpy as np

import tensorflow as tf
# from tensorflow import gfile
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.platform import tf_logging as logging

from noahnmt.utils import graph_utils


class MultikdValidationHook(tf.train.SessionRunHook):
  """Runs evaluation of a given estimator, at most every N steps.

  Note that the evaluation is done based on the saved checkpoint, which will
  usually be older than the current step.

  Can do early stopping on validation metrics if `early_stopping_rounds` is
  provided.
  """

  def __init__(self, estimator=None, input_fn=None, batch_size=None,
               eval_steps=None,
               eval_keep_best_n=0,
               eval_keep_best_metric=None,
               eval_keep_best_metric_minimize=None,
               every_n_steps=100, metrics=None, hooks=None,
               early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, 
               early_stopping_metric_delta=None, name=None,
               lang_pairs=None,
               teacher_scores=None,
               multikd_delta=0., **kargs):
    """Initializes a ValidationMonitor.

    Args:
      x: See `BaseEstimator.evaluate`.
      y: See `BaseEstimator.evaluate`.
      input_fn: See `BaseEstimator.evaluate`.
      batch_size: See `BaseEstimator.evaluate`.
      eval_steps: See `BaseEstimator.evaluate`.
      every_n_steps: Check for new checkpoints to evaluate every N steps. If a
          new checkpoint is found, it is evaluated. See `EveryN`.
      metrics: See `BaseEstimator.evaluate`.
      hooks: A list of `SessionRunHook` hooks to pass to the
        `Estimator`'s `evaluate` function.
      early_stopping_rounds: `int`. If the metric indicated by
          `early_stopping_metric` does not change according to
          `early_stopping_metric_minimize` for this many steps, then training
          will be stopped.
      early_stopping_metric: `string`, name of the metric to check for early
          stopping.
      early_stopping_metric_minimize: `bool`, True if `early_stopping_metric` is
          expected to decrease (thus early stopping occurs when this metric
          stops decreasing), False if `early_stopping_metric` is expected to
          increase. Typically, `early_stopping_metric_minimize` is True for
          loss metrics like mean squared error, and False for performance
          metrics like accuracy.
      name: See `BaseEstimator.evaluate`.

    Raises:
      ValueError: If both x and input_fn are provided.
    """
    tf.train.SessionRunHook.__init__(self)
    # TODO(mdan): Checks like this are already done by evaluate.
    if input_fn is None:
      raise ValueError("input_fn should be provided.")
    self.input_fn = input_fn
    self.batch_size = batch_size
    self.eval_steps = eval_steps
    self.metrics = metrics
    self.hooks = hooks
    self.eval_keep_best_n = eval_keep_best_n
    self.eval_keep_best_metric = eval_keep_best_metric or early_stopping_metric
    self.eval_keep_best_metric_minimize = eval_keep_best_metric_minimize or early_stopping_metric_minimize
    self.early_stopping_rounds = early_stopping_rounds
    self.early_stopping_metric = early_stopping_metric
    self.early_stopping_metric_minimize = early_stopping_metric_minimize
    self.early_stopping_metric_delta = early_stopping_metric_delta if early_stopping_metric_delta is not None else 0
    self.name = name
    self._best_value_step = None
    self._best_value = None
    self._best_metrics = None
    self._best_path = None
    self._early_stopped = False
    self._latest_path = None
    self._latest_path_step = None
    self._estimator = estimator
    self._bad_count = 0
    self._all_values = []

    if self._estimator is None:
      raise ValueError("Missing call to set_estimator.")

    self._global_step = None
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps)
    # self._sample_dir = os.path.join(self._estimator.model_dir, "samples")

    # multilingual settings
    self._language_pairs = lang_pairs.split(",")
    self._kd_flags = [True] * len(self._language_pairs)
    self._kd_placeholders = []
    self._teacher_scores = [float(x) for x in teacher_scores.split(",")]
    self._multikd_delta = float(multikd_delta)


  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()
    
    placeholders = graph_utils.get_dict_from_collection("multikd_placeholders")
    self._kd_placeholders = []
    for lang_pair in self._language_pairs:
      self._kd_placeholders.append(placeholders[lang_pair])
    # if self._sample_dir is not None:
    #   gfile.MakeDirs(self._sample_dir)

  def before_run(self, _run_context):
    feed_dict = {ph:value for ph, value in zip(self._kd_placeholders, self._kd_flags)}
    return tf.train.SessionRunArgs(self._global_step, feed_dict=feed_dict)

  def _evaluate_estimator(self):
    return self._estimator.evaluate(
        input_fn=self.input_fn, steps=self.eval_steps, hooks=self.hooks,
        name=self.name)
  

  def _eval_and_early_stopping(self, session):
    step = session.run(self._global_step)
    # Check that we are not running evaluation on the same checkpoint.
    latest_path = saver_lib.latest_checkpoint(self._estimator.model_dir)
    if latest_path is None:
      logging.debug("Skipping evaluation since model has not been saved yet "
                    "at step %d.", step)
      return False
    if latest_path is not None and latest_path == self._latest_path:
      logging.debug("Skipping evaluation due to same checkpoint %s for step %d "
                    "as for step %d.", latest_path, step,
                    self._latest_path_step)
      return False
    
    step = int(latest_path.split("-")[-1])
    self._iter_count = step
    self._timer.update_last_triggered_step(self._iter_count-1)

    self._latest_path = latest_path
    self._latest_path_step = step

    # Run evaluation and log it.
    validation_outputs = self._evaluate_estimator()
    stats = []
    for name in validation_outputs:
      if isinstance(validation_outputs[name], np.ndarray):
        validation_outputs[name] = validation_outputs[name].tolist()
      if isinstance(validation_outputs[name], list):
        value = " ".join(
            ["%s (%s)" % (str(v), name) for name, v in zip(
                ["all"] + self._language_pairs, validation_outputs[name])]
        )
      else:
        value = str(validation_outputs[name])
      stats.append("%s = %s" % (name, value))
    logging.info("Validation (step %d): %s", step, ", ".join(stats))

    self._all_values.append((step, validation_outputs))

    # reset kd_flags
    current_values = validation_outputs[self.early_stopping_metric][1:]
    self._kd_flags = [sscore < tscore + self._multikd_delta for tscore, sscore in zip (self._teacher_scores, current_values)]
    
    # Early stopping logic.
    if self.early_stopping_rounds is not None and self.early_stopping_rounds>0:
      if self.early_stopping_metric not in validation_outputs:
        raise ValueError("Metric %s missing from outputs %s." % (
            self.early_stopping_metric, set(validation_outputs.keys())))
      current_value = validation_outputs[self.early_stopping_metric]

      # take the first one if multiple
      #if isinstance(current_value, list or tuple):
      current_value = current_value[0]

      if (self._best_value is None or (self.early_stopping_metric_minimize and
                                       (current_value < self._best_value - self.early_stopping_metric_delta)) or
          (not self.early_stopping_metric_minimize and
           (current_value - self._best_value > self.early_stopping_metric_delta))):
        self._best_value = current_value
        self._best_metrics = copy.deepcopy(validation_outputs)
        self._best_value_step = step
        self._best_path = latest_path
        self._bad_count = 0
      else:
        self._bad_count += 1

      logging.info("Best step by Now: {} with {} = {}. Checkpoint path: {}"
                   .format(self._best_value_step,
                           self.early_stopping_metric, str(self._best_value),
                           self._best_path))
      stop_now = self._bad_count >= self.early_stopping_rounds #(step - self._best_value_step >= self.early_stopping_rounds)
      return stop_now
    return False


  def after_run(self, run_context, run_values):
    step = run_values.results
    self._iter_count = step
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if not self._should_trigger:
      return None

    # eval when we have a new checkpoint
    latest_path = saver_lib.latest_checkpoint(self._estimator.model_dir)
    if (latest_path is not None and 
        latest_path != self._latest_path):
      self._should_trigger = True
    else:
      self._should_trigger = False
      
    if not self._should_trigger:
      return None

    stop_now = self._eval_and_early_stopping(run_context.session)
    if stop_now:
      logging.info("Stopping. Best step: {} with {} = {}. Checkpoint path: {}"
                    .format(self._best_value_step,
                            self.early_stopping_metric, self._best_value,
                            self._best_path))

      run_context.request_stop()

    return False
  

  def end(self, session):
    # last evaluation
    _ = self._eval_and_early_stopping(session)

    if (self.eval_keep_best_n > 0 and 
        self.early_stopping_rounds is not None):
      logging.info("Saving best {} checkpoints according to {}"
                    .format(self.eval_keep_best_n, 
                            self.eval_keep_best_metric))
      # copy best_n checkpoints
      if len(self._all_values) <= self.eval_keep_best_n:
        logging.info("Saving cancelled because the number of checkpoints is not enough.")
      else:
        # sorted according to metric, descending order
        sorted_values = sorted(
            self._all_values, 
            key=lambda x: x[1][self.eval_keep_best_metric][0], 
            reverse=(not self.eval_keep_best_metric_minimize))
        # set dirs
        model_dir = self._estimator.model_dir
        output_dir = os.path.join(model_dir, "best_{}_{}".format(
            self.eval_keep_best_n, 
            self.eval_keep_best_metric))
        if tf.gfile.Exists(output_dir):
          tf.gfile.DeleteRecursively(output_dir)
        tf.gfile.MakeDirs(output_dir)

        # copy train_options
        tf.gfile.Copy(
            os.path.join(model_dir, "train_options.json"), 
            os.path.join(output_dir, "train_options.json"),
            overwrite=True)

        tf.logging.info("Saving to %s" % output_dir)
        for i, (step, values) in enumerate(sorted_values[:self.eval_keep_best_n]):
          base_name = "model.ckpt-{}".format(step)
          ckpt = os.path.join(model_dir, base_name)

          tf.logging.info("[{}] {}={} model={}".format(i, self.eval_keep_best_metric, 
                                                      str(values[self.eval_keep_best_metric]),
                                                      ckpt))
          try:
            filenames = tf.gfile.Glob(ckpt + ".*")
          except tf.errors.NotFoundError:
            filenames = tf.gfile.Glob(ckpt + ".*")
          for name in filenames:
            new_name = os.path.join(output_dir, os.path.basename(name))
            tf.gfile.Copy(name, new_name, overwrite=True)
        tf.logging.info("Done.")
