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

from __future__ import absolute_import, division, print_function
import sys
from copy import deepcopy
import pandas as pd

from dataset_base import _DEFAULT_DS_VAL_OPTIONS
from dataset_mpisintel import MPISintelDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS
import moxing as mox
import os
import shutil

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ckpt', type=str, default='./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000',
                    help='the path of checkpoint')
parser.add_argument('--obs', type=bool, default=True,
                    help='whether copy ckpt from obs')
parser.add_argument('--dataset', type=str, default='/cache/')
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)

args = parser.parse_args()

_DATASET_ROOT = args.dataset
_MPISINTEL_ROOT = os.path.join(_DATASET_ROOT, 'MPI-Sintel-complete')

os.makedirs(_MPISINTEL_ROOT)
mox.file.copy_parallel('obs://pwcnet-final/MPI-Sintel-complete', _MPISINTEL_ROOT)
if args.obs:
    mox.file.copy_parallel('obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned',
                           './pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned')

gpu_devices = ['/device:CPU:0']
controller = '/device:CPU:0'

# More options...
mode = 'val_notrain'  # 'val_notrain'            # We're doing evaluation using the entire dataset for evaluation
ckpt_path = args.ckpt  # Model to eval

# Load the dataset in evaluation mode, starting with the default evaluation options
ds_opts = deepcopy(_DEFAULT_DS_VAL_OPTIONS)
ds_opts['type'] = 'clean'
ds = MPISintelDataset(mode=mode, ds_root=_MPISINTEL_ROOT, options=ds_opts)

# Display dataset configuration
ds.print_config()

# Configure the model for evaluation, starting with the default evaluation options
nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1  # Setting this to 1 leads to more accurate evaluations of the processing time
nn_opts['use_tf_data'] = False  # Don't use tf.data reader for this simple task
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller  # Evaluate on CPU or GPU?

# We're evaluating the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and uspampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
nn_opts['adapt_info'] = (1, 436, 1024, 2)

# Instantiate the model in evaluation mode and display the model configuration
nn = ModelPWCNet(mode=mode, options=nn_opts, dataset=ds)

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops

# 导入网络模型文件
import alexnet

# 指定checkpoint路径
ckpt_path = "/opt/npu/model_ckpt/alexnet/model_8p/model.ckpt-0"


def main():
    tf.reset_default_graph()
    # 定义网络的输入节点
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # 调用网络模型生成推理图
    logits = alexnet.inference(inputs, version="he_uniform",
                               num_classes=1000, is_training=False)
    # 定义网络的输出节点
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    with tf.Session() as sess:
        # 保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')  # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',  # 传入write_graph生成的模型文件
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
            output_node_names='output',  # 与定义的推理网络输出节点保持一致
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/alexnet.pb',  # 改为需要生成的推理网络的名称
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()

# freeze_graph的关键参数解释，其他参数保持默认：
# input_graph：模型文件，通过write_graph生成的模型文件。
# input_binary：配合input_graph用，为true时，input_graph为二进制，为false时，input_graph为文件。默认值是False。
# input_checkpoint：checkpoint文件地址。
# output_node_names：输出节点的名字，有多个时用逗号分开。
# output_graph：用来保存转换后的模型输出文件，即pb文件的保存地址。
# 运行之后，./pb_model/文件夹中就会出现alexnet.pb文件，这是我们转换后的用于推理的pb图文件。
