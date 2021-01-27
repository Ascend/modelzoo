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
import os
import sys
import pathlib

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent.parent))

from networks import model
from tensorflow.python.compat import compat


def ckpt2pb(ckptpath):
    tf.reset_default_graph()
    input_images = tf.placeholder(tf.float32, shape=[1, 800, 800, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    with compat.forward_compatibility_horizon(2019, 5, 1):
        binarize_map, threshold_map, thresh_binary = model.dbnet(input_images, is_training=False)
    # binarize_map, threshold_map, thresh_binary = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    saver.restore(sess, ckptpath)

    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ['detector_layer/Sigmoid'])

    with tf.gfile.FastGFile('db_resnet.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    ckpt2pb('/home/yhm/dbgit/DB_clean/logs_adam_paperlr/ckpt/DB_resnet_v1_50_adam_model.ckpt-168000')
    
