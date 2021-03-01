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
import argparse
from typing import List

import tensorflow as tf

from common.tf_utils import ckpt_iterator
import common.utils as utils
import const
from datasets.data_wrapper_base import DataWrapperBase
from datasets.matting_data_wrapper import MattingDataWrapper
from factory.base import CNNModel
import factory.matting_nets as matting_nets
from helper.base import Base
from helper.evaluator import Evaluator
from helper.evaluator import MattingEvaluator
from metrics.base import MetricManagerBase
from factory.matting_converter import ProbConverter
from tensorflow.python.tools import freeze_graph

def parse_arguments(arguments: List[str]=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-ckpt_convert', default='../mmnet_res/results/train/MMNetModel-1593500', help='output path.')
    subparsers = parser.add_subparsers(title="Model", description="")

    # -- * -- Common Arguments & Each Model's Arguments -- * --
    CNNModel.add_arguments(parser, default_type="matting")
    matting_nets.MattingNetModel.add_arguments(parser)
    for class_name in matting_nets._available_nets:
        subparser = subparsers.add_parser(class_name)
        subparser.add_argument("--model", default=class_name, type=str, help="DO NOT FIX ME")
        add_matting_net_arguments = eval("matting_nets.{}.add_arguments".format(class_name))
        add_matting_net_arguments(subparser)

    Evaluator.add_arguments(parser)
    Base.add_arguments(parser)
    DataWrapperBase.add_arguments(parser)
    MattingDataWrapper.add_arguments(parser)
    MetricManagerBase.add_arguments(parser)

    args = parser.parse_args(arguments)

    model_arguments = utils.get_subparser_argument_list(parser, args.model)
    args.model_arguments = model_arguments

    return args

args = parse_arguments()

x = tf.placeholder(tf.float32, [1, 256, 256, 3], 'input_x')

model = matting_nets.MMNetModel(args, None)
images = model.build_images(x)
logit_scores, endpoints = model.build_inference(images, is_training=False)
output = ProbConverter.convert(logit_scores, 'output', args.num_classes)

ckpt_path = args.ckpt_convert

with tf.Session() as sess:
    # 保存图，在./pb_model文件夹中生成model.pb文件
    # model.pb文件将作为input_graph给到接下来的freeze_graph函数
    tf.train.write_graph(sess.graph_def, './offline_infer/', 'model.pb')  # 通过write_graph生成模型文件
    freeze_graph.freeze_graph(
        input_graph='./offline_infer/model.pb',  # 传入write_graph生成的模型文件
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
        output_node_names='output',  # 与定义的推理网络输出节点保持一致
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./offline_infer/mmnet.pb',  # 改为需要生成的推理网络的名称
        clear_devices=False,
        initializer_nodes='')
print("done")