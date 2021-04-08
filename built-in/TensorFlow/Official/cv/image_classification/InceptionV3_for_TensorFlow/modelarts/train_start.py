# coding=utf-8
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
import ast
import os
import glob
import argparse

import moxing as mox
import inception.data_loader as dl
import inception.model as ml
import inception.hyper_param as hp
import inception.layers as ly
import inception.logger as lg
import inception.trainer as tr
import inception.create_session as cs
import tensorflow as tf
from inception import inception_v3
from tensorflow.python.tools import freeze_graph
from tensorflow.contrib import slim as slim


def parse_args():
    """
    get arguments from ModelArts
    return:
        input arguments
    """
    parser = argparse.ArgumentParser(formatter_class =
                                     argparse.ArgumentDefaultsHelpFormatter)
    # mode output dir from obs
    parser.add_argument("--train_url", type = str, default = "obs://inception/",
                        help = "the path model saved")
    # data input dir from obs
    parser.add_argument("--data_url", type = str, default = "obs://inception/",
                        help = "the training data")
    # use default value, because ModelArts only support 1npu training
    parser.add_argument("--rank_size", default = 1, type = int,
                        help = "number of NPUs to use.")
    parser.add_argument("--network", default = "inception_v3", type = str,
                        help = "the network name")

    # mode and parameters related 
    parser.add_argument("--mode", default = "train",
                        help = "mode to run the program  e.g. train, evaluate, "
                             "and train_and_evaluate")
    parser.add_argument("--max_train_steps", default = 100, type = int,
                        help = "max steps to train")
    parser.add_argument("--iterations_per_loop", default = 10, type = int,
                        help = "the number of steps in devices for each iteration")
    parser.add_argument("--max_epochs", default = 5, type = int,
                        help = "total epochs for training")
    parser.add_argument("--epochs_between_evals", default = 5, type = int,
                        help = "the interval between train and evaluation, only"
                             " meaningful when the mode is train_and_evaluate")

    parser.add_argument("--data_dir", default = "path/data",
                        help = "directory of dataset.")
    # path for evaluation
    parser.add_argument("--eval_dir", default = "/cache/model",
                        help = "directory to evaluate.")
    parser.add_argument("--dtype", default = tf.float32,
                        help = "data type of inputs.")
    parser.add_argument("--use_nesterov", default = True, type = ast.literal_eval,
                        help = "whether to use Nesterov in optimizer")
    parser.add_argument("--label_smoothing", default = 0.1, type = float,
                        help = "label smoothing factor")
    parser.add_argument("--weight_decay", default = 0.00001,
                        help = "weight decay for regularization")
    parser.add_argument("--batch_size", default = 128, type = int,
                        help = "batch size for one NPU")

    # learning rate for every step
    parser.add_argument("--lr", default = 0.045, type = float,
                        help = "initial learning rate")
    parser.add_argument("--lr_decay", default = 0.8, type = float,
                        help = "learning rate decay")
    parser.add_argument("--lr_decay_steps", default = 10000, type = int,
                        help= "learning rate decay steps")
    parser.add_argument("--T_max", default = 100, type = int,
                        help = "T_max for cosing_annealing learning rate")
    parser.add_argument("--momentum", default = 0.9, type = float,
                        help = "momentum used in optimizer.")
    # display frequency
    parser.add_argument("--display_every", default = 100, type = int,
                        help = "the frequency to display info")
    # log file
    parser.add_argument("--log_dir", default = "/cache/model",
                        help = "log directory")
    parser.add_argument("--log_name", default = "inception_v3.log",
                        help = "name of log file")

    parser.add_argument("--restore_path", default = "",
                        help = "restore path")

    parser.add_argument('--num_classes', default = 1000, type = int,
                        help="the number class of dataset")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


def frozen_model(args):
    """
    frozen the model of ckpt
    params:
        args: type is dict, include ckpt_path, output_graph
    """
    tf.reset_default_graph()
    # modify input node
    inputs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="input")
    # build inference graph
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        top_layer, end_points = inception_v3.inception_v3(inputs=inputs,
                                                          num_classes=args.get('num_classes'),
                                                          dropout_keep_prob=1.0,
                                                          is_training = False)
    logits = top_layer
    logits = tf.cast(logits, tf.float32)

    with tf.Session() as sess:
        # save unfrozen graph
        tf.train.write_graph(sess.graph_def, '/cache/model', 'model.pb')
        # start to froze graph
        freeze_graph.freeze_graph(
		        input_graph='/cache/model/model.pb',
		        input_saver='',
		        input_binary=False,
		        input_checkpoint=args.get('ckpt_path'),
		        output_node_names='InceptionV4/Logits/Logits/BiasAdd',
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph=args.get('output_graph'),
		        clear_devices=False,
		        initializer_nodes='')
    print("done")


def main():
    """
    1.prepare dataset
    2.train and evaluate
    3.frozen the ckpt model
    4.copy log and obs to obs storage
    """
    input_args = parse_args()
    data_path = "/cache/data"
    model_output_path = "/cache/model"

    if not os.path.exists(data_path):
        os.makedirs(data_path, 0o755)
    mox.file.copy_parallel(input_args.data_url, data_path)

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path, 0o755)

    # pre_train model path
    pretrained_model_dir = os.path.join(data_path, "ckpt",
                                        input_args.restore_path)
    input_args.restore_path = pretrained_model_dir

    input_args.global_batch_size = input_args.batch_size * input_args.rank_size
    input_args.data_dir = data_path
    session = cs.CreateSession()
    data = dl.DataLoader(input_args)
    hyper_param = hp.HyperParams(input_args)
    layers = ly.Layers()
    logger = lg.LogSessionRunHook(input_args)
    model = ml.Model(input_args, data, hyper_param, layers, logger)
    trainer = tr.Trainer(session, input_args, data, model, logger)

    if input_args.mode == "train":
        trainer.train()
    elif input_args.mode == "evaluate":
        trainer.evaluate()
    elif input_args.mode == "train_and_evaluate":
        print("=======start train_and_evaluate=============")
        trainer.train_and_evaluate()
    else:
        raise ValueError("Invalid mode.")

    ckpt_list = glob.glob("/cache/model/model.ckpt-*.meta")
    if not ckpt_list:
        print("ckpt file not generated.")
        return
    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1].rsplit(".", 1)[0]
    print("====================%s" % ckpt_model)
    frozen_args = {'num_classes': input_args.num_classes,
                   'ckpt_path': ckpt_model,
                   'output_graph': '/cache/model/inception_v3_tf.pb'}

    frozen_model(frozen_args)
    # copy model to train_url
    mox.file.copy_parallel("/cache/model", input_args.train_url)


if __name__ == "__main__":
    main()

