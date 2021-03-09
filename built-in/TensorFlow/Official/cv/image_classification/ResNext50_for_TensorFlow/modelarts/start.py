# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import sys
import ast
import os
import argparse
import glob
import moxing as mox
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from npu_bridge.estimator import npu_ops
from utils import create_session as cs
from utils import logger as lg
from data_loader.resnet50 import data_loader as dl
from models.resnet50 import resnet, res50_helper
from models.resnet50 import res50_model as ml
from optimizers import optimizer as op
from losses import res50_loss as ls
from trainers import gpu_base_trainer as tr
# from configs import res50_config as cfg
from hyper_param import hyper_param as hp
from layers import layers as ly

OUTPUT_PATH = "/cache/model"
DATA_PATH = "/cache/data"


def set_env():
    """
    set environment of DEVICE_INDEX
    """
    os.environ['DEVICE_INDEX'] = os.getenv('RANK_ID')


def args_parser():
    """
    get super parameter
    return:
        parser_args
    """
    parser = argparse.ArgumentParser(description="train resnet50")

    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')
    parser.add_argument('--config_file', type=str, default='res50_32bs_1p_host',
                        help='the config file')
    parser.add_argument('--max_train_steps', type=int, default=10000,
                        help='max_train_steps')
    parser.add_argument('--iterations_per_loop', default=1000,
                         help='iterations config used.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='num_classes')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='num_epochs')
    parser.add_argument('--learning_rate_maximum', type=float, default=0.1,
                        help='learning_rate_maximum')
    parser.add_argument('--debug', default=True, type=ast.literal_eval,
                        help='debug mode config used.')
    parser.add_argument('--eval', default=False, type=ast.literal_eval,
                        help='evaluate config used.')
    parser.add_argument('--model_dir', default="/cache/model",
                        help='model dir path config used.')
    parser.add_argument('--restore_path', type=str, default='',
                        help='restore ckpt path')
    parser_args, _ = parser.parse_known_args()
    return parser_args


def set_config(args):
    """
    get config from file and reset the config by super parameter
    """
    configs = 'configs'
    cfg = getattr(__import__(configs, fromlist=[args.config_file]),
                  args.config_file)
    config = cfg.res50_config()
    config['data_url'] = DATA_PATH

    config['log_dir'] = OUTPUT_PATH
    config['model_dir'] = OUTPUT_PATH
    config['ckpt_dir'] = OUTPUT_PATH

    # set param from parse
    config['iterations_per_loop'] = int(args.iterations_per_loop)
    config['max_train_steps'] = int(args.max_train_steps)
    config['debug'] = args.debug
    config['eval'] = args.eval
    config['model_dir'] = args.model_dir
    config['batch_size'] = args.batch_size
    config['global_batch_size'] = config['batch_size'] * config['rank_size']
    config['num_classes'] = args.num_classes
    config['num_epochs'] = args.num_epochs
    config['learning_rate_maximum'] = args.learning_rate_maximum
    config['restore_path'] = os.path.join(DATA_PATH, "ckpt",
                                          input_args.restore_path)

    print("iterations_per_loop    :%d" % (config['iterations_per_loop']))
    print("max_train_steps        :%d" % (config['max_train_steps']))
    print("debug                  :%s" % (config['debug']))
    print("eval                   :%s" % (config['eval']))
    print("model_dir              :%s" % (config['model_dir']))
    print("batch_size             :%d" % (config['batch_size']))
    if config['num_epochs']:
        print("num_epochs         :%d" % (config['num_epochs']))
    print("learning_rate_maximum  :%f" % (config['learning_rate_maximum']))
    print("num_classes            :%d" % (config['num_classes']))
    print("restore_path           :%s" % (config['restore_path']))

    return config


def train(args):
    """
    training and generate the ckpt model
    """
    config = set_config(args)
    Session = cs.CreateSession(config)
    data = dl.DataLoader(config)
    hyper_param = hp.HyperParams(config)
    layers = ly.Layers()
    optimizer = op.Optimizer(config)
    loss = ls.Loss(config)
    # add tensorboard summary
    logger = lg.LogSessionRunHook(config)

    # get the model
    model = ml.Model(config, data, hyper_param, layers, optimizer, loss, logger)
    # use Estimator to build training process
    trainer = tr.GPUBaseTrain(Session, config, data, model, logger)

    if config['mode'] == 'train':
        trainer.train()
        if config['eval']:
            trainer.evaluate()
    elif config['mode'] == 'evaluate':
        trainer.evaluate()
    elif config['mode'] == 'train_and_evaluate':
        trainer.train_and_evaluate()
    else:
        raise ValueError('Invalid type of mode')


def model_trans(args):
    """
    frozen the model
    """
    ckpt_list = glob.glob("/cache/model/model.ckpt-*.meta")
    if not ckpt_list:
        print("ckpt file not generated.")
        return
    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1].rsplit(".", 1)[0]
    print("====================%s" % ckpt_model)

    tf.reset_default_graph()
    # set inputs node
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # create inference graph
    with res50_helper.custom_getter_with_fp16_and_weight_decay(dtype=tf.float32,
                                                               weight_decay=0.0001):
        builder = resnet.LayerBuilder(tf.nn.relu, 'channels_last', False,
                                      use_batch_norm=True,
                                      conv_initializer=None,
                                      bn_init_mode='adv_bn_init',
                                      bn_gamma_initial_value=1.0)
        top_layer = resnet.inference_resnext_impl(builder, inputs, [3, 4, 6, 3],
                                                  "original", args.num_classes)

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, '/cache/model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='/cache/model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_model,
            output_node_names='fp32_vars/final_dense',  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='/cache/model/resnext50_tf_910.pb',  # graph outputs name
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    set_env()

    input_args = args_parser()

    # copy dataset from obs to container
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, 0o755)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, 0o755)
    mox.file.copy_parallel(input_args.data_url, DATA_PATH)

    # set level of logging
    tf.logging.set_verbosity(tf.logging.INFO)

    train(input_args)

    # trans ckpt model to pb
    model_trans(input_args)

    # after train, copy log and model from container to obs
    mox.file.copy_parallel(OUTPUT_PATH, input_args.train_url)
