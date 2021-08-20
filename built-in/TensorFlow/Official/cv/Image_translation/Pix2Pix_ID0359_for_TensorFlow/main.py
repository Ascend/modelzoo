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

from npu_bridge.npu_init import *
import argparse
import os
import scipy.misc
import numpy as np
from model import pix2pix
import tensorflow as tf
parser = argparse.ArgumentParser(description='')
parser.add_argument('--blacklist_path', dest='blacklist_path', default='./', help='path of the blacklist')
parser.add_argument('--data_path', dest='data_path', default='./datasets/facades', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_mix_precision', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--autotune', dest='autotune', default='False', help='whether to enable autotune, default is False')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=100000000.0, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
args = parser.parse_args()

def main(_):
    if (not os.path.exists(args.checkpoint_dir)):
        os.makedirs(args.checkpoint_dir)
    if (not os.path.exists(args.sample_dir)):
        os.makedirs(args.sample_dir)
    if (not os.path.exists(args.test_dir)):
        os.makedirs(args.test_dir)
    config = tf.ConfigProto()  # 如果没有tf.ConfigProto，需要手工添加该行
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    blacklist_path = os.path.join(args.blacklist_path, "ops_info.json")
    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(blacklist_path)
    if args.data_dump_flag.strip()=="True":
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(args.data_dump_step)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if args.over_dump.strip()=="True":
        # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
        # enable_dump_debug：是否开启溢出检测功能
        custom_op.parameter_map["enable_dump_debug"].b = True
        # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    if args.profiling.strip()=="True":
        custom_op.parameter_map["profiling_mode"].b = False
        profilingvalue=('{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}' %(args.profiling_dump_path))
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profilingvalue)

    #  init autotune module start
    autotune = False
    autotune = os.environ.get('autotune')
    if autotune:
        autotune = autotune.lower()
        if autotune == 'true':
            print("Autotune module is :" + autotune)
            print("Autotune module has been initiated!")
            custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
        else:
            print("Autotune module is :" + autotune)
            print("Autotune module is enabled or with error setting.")
    else:
        print("Autotune module de_initiate!Pass")
    #  init autotune module end

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size, output_size=args.fine_size, dataset_name=args.dataset_name, checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
        if (args.phase == 'train'):
            model.train(args)
        else:
            model.test(args)
if (__name__ == '__main__'):
    tf.app.run()
