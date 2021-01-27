
# Copyright 2019 Huawei Technologies Co., Ltd
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
from tensorflow_core.python import pywrap_tensorflow

sess = tf.Session()
check_point_path = './tfcheck_1608121226.1290338'
saver = tf.train.import_meta_graph('params_cifar.ckpt.meta')

saver.restore(sess, check_point_path+'/params_cifar.ckpt')

graph = tf.get_default_graph()

saver = tf.train.import_meta_graph('variables/save_variables.ckpt.meta')

r=pywrap_tensorflow.NewCheckpointReader('./tfcheck_1608121226.1290338/params_cifar.ckpt')
r.get_variable_to_dtype_map()
print('end')