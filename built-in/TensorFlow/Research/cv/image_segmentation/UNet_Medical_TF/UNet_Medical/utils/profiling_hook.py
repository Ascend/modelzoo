# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import time

import numpy as np
import tensorflow as tf
#import horovod.tensorflow as hvd
import os

from utils.parse_results import process_performance_stats


class ProfilingHook(tf.estimator.SessionRunHook):
    print()
    rank_size = int(os.getenv('RANK_SIZE'))
    rank_id = int(os.getenv('DEVICE_INDEX'))

    def __init__(self, logger, batch_size, log_every, warmup_steps, mode):
        self._log_every = log_every
        self._warmup_steps = warmup_steps
        self._current_step = 0
        self._global_batch_size = batch_size * rank_size()
        self._t0 = 0
        self._timestamps = []
        self.logger = logger
        self.mode = mode

    def before_run(self, run_context):
        if self._current_step > self._warmup_steps:
            self._t0 = time.time()
            self.count = 0

    def after_run(self,
                  run_context,
                  run_values):
        if self._current_step > self._warmup_steps:
            self._timestamps.append(time.time() - self._t0)
        self._current_step += 1
        self.count += 1
        batch_time = time.time() - self.t0
        self.elapsed_secs += batch_time
        dt = self.elapsed_secs / self.count
        img_per_sec = self.global_batch_size * self.iterations_per_loop / dt
        self.logger.info('step:%6i FPS:%7.1f' %
                         (self._current_step, img_per_sec))

    def begin(self):
        pass

    def end(self, session):
        if rank_id == 0:
            throughput_imgps, latency_ms = process_performance_stats(np.array(self._timestamps),
                                                                     self._global_batch_size)
            self.logger.log(step=(),
                            data={'throughput_{}'.format(self.mode): throughput_imgps,
                                  'latency_{}'.format(self.mode): latency_ms})
