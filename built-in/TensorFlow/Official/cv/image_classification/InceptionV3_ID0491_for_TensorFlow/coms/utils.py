#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
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
# Copyright 2021 Huawei Technologies Co., Ltd
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

import platform

import tensorflow as tf
from  tensorflow.python import pywrap_tensorflow as pyw
from tensorflow.python.client import device_lib as _device_lib

def isLinuxSys():
    if platform.system() == "Linux":
        return True
    return False

def isWinSys():
    if platform.system() == "Windows":
        return True
    return False

def isHasCpu():
    info = _device_lib.list_local_devices()
    for dev in info:
        if 'CPU' in dev.name:
            return True
    return False

def isHasGpu():
    info = _device_lib.list_local_devices()

    for dev in info:
        # print(dev.name)
        if 'GPU' in dev.name:
            return True
    return False

if __name__ == '__main__':
    # path = ''
    # model_name = 'YOLO_small.ckpt'
    # if isLinuxSys():
    #     path = '/home/zhuhao/DataSets/YOLO/v1model/' + model_name
    #
    # # saver = tf.train.Saver(max_to_keep=1)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     # model_file = tf.train.latest_checkpoint(path)
    #     reader = pyw.NewCheckpointReader(path)
    #     var_to_shape_map = reader.get_variable_to_shape_map()
    #     from  pprint import pprint
    #     pprint(var_to_shape_map)
    print(isHasGpu())


