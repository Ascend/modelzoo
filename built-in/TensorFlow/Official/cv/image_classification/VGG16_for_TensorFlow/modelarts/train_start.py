# coding=utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import ast
import os

import moxing as mox
import tensorflow as tf
import vgg16.create_session as cs
import vgg16.data_loader as dl
import vgg16.hyper_param as hp
import vgg16.layers as ly
import vgg16.logger as lg
import vgg16.model as ml
import vgg16.trainer as tr
from vgg16 import vgg
from tensorflow.python.tools import freeze_graph

_CACHE_DATA_URL = "/cache/data_url"
_CACHE_TRAIN_URL = "/cache/train_url"


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_url', type=str, default='',
                        help='the dataset path')
    parser.add_argument('--train_url', type=str, default='',
                        help='the  path model saved')

    parser.add_argument('--rank_size', default=1, type=int,
                        help='number of NPUs  to use.')

    # mode and parameters related
    parser.add_argument('--mode', default='train',
                        choices=["train", "evaluate", "train_and_evaluate"],
                        help='mode to run the program')
    parser.add_argument('--max_train_steps', default=100, type=int,
                        help='max steps to train')
    parser.add_argument('--iterations_per_loop', default=10, type=int,
                        help='the number of steps in devices for each '
                             'iteration')
    parser.add_argument('--max_epochs', default=None, type=int,
                        help='total epochs for training')
    parser.add_argument('--epochs_between_evals', default=5, type=int,
                        help='the interval between train and evaluation, only '
                             'meaningful when the mode is train_and_evaluate')

    # dataset
    parser.add_argument('--data_dir', default='path/data',
                        help='directory of dataset.')

    # path for evaluation
    parser.add_argument('--eval_dir', default='path/eval',
                        help='directory to evaluate.')

    parser.add_argument('--dtype', default=tf.float32,
                        help='data type of inputs.')
    parser.add_argument('--use_nesterov', default=True, type=ast.literal_eval,
                        help='whether to use Nesterov in optimizer')
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing factor')
    parser.add_argument('--weight_decay', default=0.0001,
                        help='weight decay for regularization')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size for one NPU')

    # learning rate and momentum
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--T_max', default=150, type=int,
                        help='T_max for cosing_annealing learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum used in optimizer.')

    # display frequency
    parser.add_argument('--display_every', default=1, type=int,
                        help='the frequency to display info')

    # log file
    parser.add_argument('--log_name', default='vgg16.log',
                        help='name of log file')
    parser.add_argument('--log_dir', default=_CACHE_TRAIN_URL,
                        help='log directory')
    parser.add_argument('--restore_path', default='',
                        help="""restore path""")
    parser.add_argument('--restore_exclude', default=['dense_2'],
                        type=ast.literal_eval, help="""restore_exclude""")
    parser.add_argument('--class_num', default=1000, type=int,
                        help="the class num")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def _get_last_ckpt(ckpt_dir):
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.startswith('model.ckpt-')
                  and ckpt_file.endswith('.index')]
    ckpt_files = [ckpt_file.split('.index')[0] for ckpt_file in ckpt_files]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _frozen_graph(args):
    ckpt_file = _get_last_ckpt(args.log_dir)
    if not ckpt_file:
        return

    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                            name="input")
    top_layer = vgg.vgg_impl(inputs, False, args.class_num)
    # create inference graph
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_file,
            output_node_names='dense_2/BiasAdd',  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(args.log_dir, 'vgg16_tf_910.pb'),
            clear_devices=False,
            initializer_nodes='')
    print("frozen graph done")


def _main():
    args = _parse_args()
    args.data_dir = _CACHE_DATA_URL
    os.environ['DEVICE_INDEX'] = "0"
    os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
    os.makedirs(_CACHE_DATA_URL, exist_ok=True)
    mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)

    args.global_batch_size = args.batch_size * args.rank_size

    session = cs.CreateSession()
    data = dl.DataLoader(args)
    hyper_param = hp.HyperParams(args)
    layers = ly.Layers()
    logger = lg.LogSessionRunHook(args)
    model = ml.Model(args, data, hyper_param, layers, logger)

    trainer = tr.Trainer(session, args, data, model, logger)

    if args.restore_path:
        args.restore_path = os.path.join(
            _CACHE_DATA_URL, args.restore_path)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'evaluate':
        trainer.evaluate()
    elif args.mode == 'train_and_evaluate':
        trainer.train_and_evaluate()
    else:
        raise ValueError("Invalid mode.")
    _frozen_graph(args)
    mox.file.copy_parallel(args.log_dir, args.train_url)


if __name__ == '__main__':
    _main()
