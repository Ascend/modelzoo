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
import os
import sys

import argparse
from absl import logging

import mindspore.common.dtype as mstype
from mindspore.dataset.engine import GeneratorDataset
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
import mindspore as ms
from mindspore import Tensor, context, Model
from mindspore.nn import TrainOneStepCell, WithLossCell, SoftmaxCrossEntropyWithLogits

from src.callbacks import LossCallBack, EvalCallBack
import src.constants as rconst
from src.dataset import create_dataset
from src.metrics import NCFMetric
import src.movielens
from src.ncf import NCFModel, NetWithLossClass, TrainStepWrap, PredictWithSigmoid


logging.set_verbosity(logging.INFO)


def argparse_init():
    parser = argparse.ArgumentParser(description='NCF')

    parser.add_argument("--data_path", type=str, default="./dataset/")  # The location of the input data.
    parser.add_argument("--dataset", type=str, default="ml-1m")  # Dataset to be trained and evaluated. ["ml-1m", "ml-20m"]
    parser.add_argument("--train_epochs", type=int, default=14)  # The number of epochs used to train.
    parser.add_argument("--batch_size", type=int, default=256)  # Batch size for training and evaluation
    parser.add_argument("--num_neg", type=int, default=4)  # The Number of negative instances to pair with a positive instance.
    parser.add_argument("--layers", type=int, default=[64, 32, 16])  # The sizes of hidden layers for MLP
    parser.add_argument("--num_factors", type=int, default=16)  # The Embedding size of MF model.
    parser.add_argument("--output_path", type=str, default="./output/")  # The location of the output file.
    parser.add_argument("--loss_file_name", type=str, default="loss.log")  # Loss output file.
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/")  # The location of the checkpoint file.
    return parser


def test_train():
    parser = argparse_init()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    layers = args.layers
    num_factors = args.num_factors
    epochs = args.train_epochs

    ds_train, num_train_users, num_train_items = create_dataset(test_train=True, data_dir=args.data_path, dataset=args.dataset,
                                                                train_epochs=args.train_epochs, batch_size=args.batch_size, num_neg=args.num_neg)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    ncf_net = NCFModel(num_users=num_train_users,
                       num_items=num_train_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)
    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net)

    train_net.set_train()

    model = Model(train_net)
    callback = LossCallBack(loss_file_path=os.path.join(args.output_path, args.loss_file_name ))
    ckpt_config = CheckpointConfig(save_checkpoint_steps=(4970845+args.batch_size-1)//(args.batch_size), keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix='NCF', directory=args.checkpoint_path, config=ckpt_config)
    model.train(epochs, 
                ds_train, 
                callbacks=[callback, ckpoint_cb], 
                dataset_sink_mode=True)


if __name__ == '__main__':
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Davinci",
                        save_graphs=True,
                        device_id=devid)

    test_train()
