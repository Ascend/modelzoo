# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Executes Keras benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark


class KerasBenchmark(PerfZeroBenchmark):
  """Base benchmark class with methods to simplify testing."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               flag_methods=None,
               tpu=None):
    assert tf.version.VERSION.startswith('2.')
    super(KerasBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods,
        tpu=tpu)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        top_1_max=None,
                        top_1_min=None,
                        log_steps=None,
                        total_batch_size=None,
                        warmup=1,
                        start_time_sec=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      top_1_max: highest passing level for top_1 accuracy.
      top_1_min: lowest passing level for top_1 accuracy.
      log_steps: How often the log was created for stats['step_timestamp_log'].
      total_batch_size: Global batch-size.
      warmup: number of entries in stats['step_timestamp_log'] to ignore.
      start_time_sec: the start time of the program in seconds since epoch
    """

    metrics = []
    if 'accuracy_top_1' in stats:
      metrics.append({'name': 'accuracy_top_1',
                      'value': stats['accuracy_top_1'],
                      'min_value': top_1_min,
                      'max_value': top_1_max})
      metrics.append({'name': 'top_1_train_accuracy',
                      'value': stats['training_accuracy_top_1']})

    if (warmup and 'step_timestamp_log' in stats and
        len(stats['step_timestamp_log']) > warmup):
      # first entry in the time_log is start of step 1. The rest of the
      # entries are the end of each step recorded
      time_log = stats['step_timestamp_log']
      elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
      num_examples = (
          total_batch_size * log_steps * (len(time_log) - warmup - 1))
      examples_per_sec = num_examples / elapsed
      metrics.append({'name': 'exp_per_second',
                      'value': examples_per_sec})

    if 'avg_exp_per_second' in stats:
      metrics.append({'name': 'avg_exp_per_second',
                      'value': stats['avg_exp_per_second']})

    if start_time_sec and 'step_timestamp_log' in stats:
      time_log = stats['step_timestamp_log']
      # time_log[0] is recorded at the beginning of the first step.
      startup_time = time_log[0].timestamp - start_time_sec
      metrics.append({'name': 'startup_time', 'value': startup_time})

    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=-1,
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_str})
