# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

import tensorflow as tf
import os,sys
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
from npu_bridge.npu_init import *

class CreateSession():
    def __init__(self):
        self.estimator_config = tf.ConfigProto(
            inter_op_parallelism_threads=10,
            intra_op_parallelism_threads=10,
            allow_soft_placement=True)

        profiling_options = '{"output":"/home/etp_output","task_trace":"on"}'
        self.profiling_config = ProfilingConfig(enable_profiling=True, profiling_options=profiling_options)

        self.estimator_config.gpu_options.allow_growth = True

        #---------------add--------------
        custom_op = self.estimator_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["hcom_parallel"].b = True
        self.estimator_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        self.estimator_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        custom_op.parameter_map["graph_run_mode"].i = 0
        #--------------------------------

        self.set_env()

    def set_env(self):
        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

