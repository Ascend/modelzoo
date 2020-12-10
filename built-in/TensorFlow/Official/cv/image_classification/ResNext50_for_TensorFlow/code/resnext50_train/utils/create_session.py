# coding=utf-8
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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import tensorflow as tf
import os,sys

class CreateSession():
    def __init__(self, config): 
        self.config = config

        if self.config['accelerator'] == '1980':
            from tensorflow.python.client import device_lib
            #from tensorflow.contrib.offline_train.python import npu_ops
            from npu_bridge.estimator import npu_ops
            #self.estimator_config = tf.ConfigProto(allow_soft_placement=True, min_group_size=20, use_off_line=True)
            self.estimator_config = tf.ConfigProto(allow_soft_placement=True)
            custom_op = self.estimator_config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["min_group_size"].b = 20
        else:
            self.estimator_config = tf.ConfigProto(allow_soft_placement=False)

        self.estimator_config.gpu_options.allow_growth = True

        if self.config['accelerator'] == '1980':
            local_device_protos = device_lib.list_local_devices(self.estimator_config)

        self.set_env()
      

    def set_env(self):
        # TODO, get env from config file
        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        # barrier = self.hvd.allreduce(tf.constant(0, dtype=tf.float32))
        # tf.Session(config=self.estimator_config).run(barrier)


    def get_config(self):
        self.estimator_config.gpu_options.visible_device_list = str(0)
#        self.estimator_config.gpu_options.force_gpu_compatible = True  # Force pinned memory
        self.estimator_config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
        self.estimator_config.inter_op_parallelism_threads = 5
        return self.estimator_config


