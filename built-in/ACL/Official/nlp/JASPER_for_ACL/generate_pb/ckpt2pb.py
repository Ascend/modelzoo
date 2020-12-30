# convert checkpoint 2 pb

# -*- coding: utf-8 -*-
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.summary import FileWriter
from tensorflow.python.tools import freeze_graph

def freeze_graph_fun(input_pbtxt,output_pb,checkpoint): 
  freeze_graph.freeze_graph(
        input_graph = input_pbtxt,
        input_saver = '',
        input_binary=False,
        input_checkpoint=checkpoint,
        output_node_names='ForwardPass/fully_connected_ctc_decoder/fully_connected/BiasAdd',
        clear_devices=True,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph = output_pb,
        initializer_nodes='',
        )



if __name__ == '__main__':
  #checkpoint输入路径
  #input_checkpoint='jasper_log_folder/logs/model.ckpt-52800'
  inputpath='./jasper_log_folder/logs-5x3/model.ckpt-7700'
  #pb输出路径
  out_pb_path="jasper_infer_float32.pb"
  #调用freeze_graph将ckpt转为pb
  #freeze_graph_fun1(inputpath,out_pb_path)
  freeze_graph_fun("./graph/graph_jasper_infer_ori.pbtxt",out_pb_path,inputpath)
  #printTensorName(input_checkpoint)


