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
from mindspore.train.serialization import load_checkpoint, load_param_into_net
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
    parser.add_argument("--eval_batch_size", type=int, default=160000)  # The batch size used for evaluation.
    parser.add_argument("--layers", type=int, default=[64, 32, 16])  # The sizes of hidden layers for MLP
    parser.add_argument("--num_factors", type=int, default=16)  # The Embedding size of MF model.
    parser.add_argument("--output_path", type=str, default="./output/")  # The location of the output file.
    parser.add_argument("--eval_file_name", type=str, default="eval.log")  # Eval output file.
    parser.add_argument("--checkpoint_file_path", type=str, default="./checkpoint/NCF-14_19418.ckpt")  # The location of the checkpoint file.
    return parser


def test_eval():
    parser = argparse_init()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    layers = args.layers
    num_factors = args.num_factors
    topk = rconst.TOP_K
    num_eval_neg = rconst.NUM_EVAL_NEGATIVES

    ds_eval, num_eval_users, num_eval_items = create_dataset(test_train=False, data_dir=args.data_path, dataset=args.dataset,
                                                             train_epochs=args.train_epochs, eval_batch_size=args.eval_batch_size)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    ncf_net = NCFModel(num_users=num_eval_users,
                       num_items=num_eval_items,
                       num_factors=num_factors,
                       model_layers=layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)
    param_dict = load_checkpoint(args.checkpoint_file_path)
    #import pdb;pdb.set_trace()
    load_param_into_net(ncf_net, param_dict)

    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net)
    train_net.set_train()
    eval_net = PredictWithSigmoid(ncf_net, topk, num_eval_neg)

    ncf_metric = NCFMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"ncf": ncf_metric})

    ncf_metric.clear()
    out = model.eval(ds_eval)

    eval_file_path=os.path.join(args.output_path, args.eval_file_name)
    eval_file = open(eval_file_path, "a+")
    eval_file.write("EvalCallBack: HR = {}, NDCG = {}\n".format(out['ncf'][0], out['ncf'][1]))
    eval_file.close()
    print("EvalCallBack: HR = {}, NDCG = {}".format(out['ncf'][0], out['ncf'][1]))


if __name__ == '__main__':
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Davinci",
                        save_graphs=True,
                        device_id=devid)

    test_eval()
