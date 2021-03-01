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
import os 

import tensorflow as tf
import sys
sys.path.append('.')

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

def convert_data(args):
    session = tf.Session(config=const.TF_SESSION_CONFIG)
    dataset_names = args.dataset_split_name

    dataset = MattingDataWrapper(
        args,
        session,
        dataset_names[0],
        is_training=False,
    )
    _, _, images, masks = dataset.get_input_and_output_op()
    iters = dataset.num_samples // args.batch_size
    
    output_path = args.output
    os.makedirs(os.path.join(output_path, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test/masks'), exist_ok=True)

    for i in range(iters):
        [img_np, msk_np] = session.run([images, masks])
        path_img = os.path.join(output_path, "test/images/%d.bin" % i)
        path_msk = os.path.join(output_path, "test/masks/%d.bin" % i)
        img_np.tofile(path_img)
        msk_np.tofile(path_msk)

def parse_arguments(arguments: List[str]=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default='offline_infer/Bin', help='output path.')
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


if __name__ == "__main__":
    args = parse_arguments()
    log = utils.get_logger("MattingEvaluator", None)

    log.info(args)
    convert_data(args)
