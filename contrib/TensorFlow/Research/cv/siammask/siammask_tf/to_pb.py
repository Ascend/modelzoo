# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
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
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
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

# -------------------------------------------------------------------------------

import tensorflow as tf

from model.custom import siammask
from tensorflow.python.compat import compat


def ckpt2pb():
    ckptpath = "/home/hagongda/zx/20210120/mySiamMask/logs1/ckpt/siamMask_model.ckpt-36000"
    tf.reset_default_graph()
    template_size = 127
    search_size = 255
    batch_size = 1
    template = tf.placeholder(tf.float32, shape=[batch_size, template_size, template_size, 3], name='template')
    search = tf.placeholder(tf.float32, shape=[batch_size, search_size, search_size, 3], name='search')

    with tf.name_scope('model') as scope:
        with compat.forward_compatibility_horizon(2019, 5, 1):
            rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature, search_feature = siammask(template, search,
                                                                                                   False)
    print(rpn_pred_cls.name, rpn_pred_loc.name, rpn_pred_mask.name)
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    saver.restore(sess, ckptpath)

    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ['model/siammask/rpn_model/conv2d_3/BiasAdd',
         "model/siammask/rpn_model/conv2d_7/BiasAdd",
         "model/siammask/mask_model/conv2d_3/BiasAdd"])

    with tf.gfile.FastGFile('siammask.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    print("end")


if __name__ == '__main__':
    ckpt2pb()
