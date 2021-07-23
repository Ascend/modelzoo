# -*- coding: utf-8 -*-

# Copyright 2020 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at# 
# 
#     http://www.apache.org/licenses/LICENSE-2.0# 
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import random
import argparse
import configparser

import numpy as np
import pandas as pd

import torch


from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


parser = argparse.ArgumentParser(description='Wide&Deep')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training.')
parser.add_argument('--device_id', default=0, type=int, help='device id')
parser.add_argument('--dist', default=False, action='store_true', help='8p distributed training')
parser.add_argument('--device_num', default=1, type=int,
                    help='num of npu device for training')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=1024, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O1', type=str,
                    help='apex opt level')
parser.add_argument('--data_path', required=True, type=str, help='train data, and is to be')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size for training and testing')
parser.add_argument('--epochs', default=3, type=int, help='epochs for training')

parser.add_argument('--steps', default=0, type=int, help='steps for training')

TOTAL_TRAIN_VAL_SAMPLE = int(45840616 * 0.9)
TOTAL_TEST_SAMPLE = int(45840616 * 0.1)
def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    fix_random(args.seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']
    col_names = target + dense_features + sparse_features
    nrows = TOTAL_TRAIN_VAL_SAMPLE // args.device_num
    if args.device_num > 1:
        skiprows = list(range(1, 1 + args.device_id * nrows))
    else:
        skiprows = None

    # 2.count #unique features for each sparse field,and record dense feature field name
    start_time = time.time()
 
    data_trainval = pd.read_csv(args.data_path + '/wdl_trainval.txt', sep='\t', skiprows=skiprows, nrows=nrows)
    data_test = pd.read_csv(args.data_path + '/wdl_test.txt', sep='\t')
    print('Data loaded in {}s'.format(time.time() - start_time))

    sparse_nunique = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
    fixlen_feature_columns = [SparseFeat(feat, sparse_nunique[idx], embedding_dim=4)
                              for idx, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    print(fixlen_feature_columns)

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    print('Generating input data for model...')
    start_time = time.time()
    train, test = data_trainval, data_test
    train_model_input = {name: train[name].astype(float) for name in feature_names}
    test_model_input = {name: test[name].astype(float) for name in feature_names}
    print('Input data generated in {}s'.format(time.time() - start_time))

    # 4.Define Model,train,predict and evaluate
    if args.dist:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29680'
        torch.distributed.init_process_group(backend='hccl', world_size=args.device_num, rank=args.device_id)
        print('distributed train enabled')

    device = 'npu:' + str(args.device_id)
    torch.npu.set_device(device)
    print('train on: ', device)

    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary', dnn_hidden_units=(256, 256, 256), dnn_dropout=0.5, device=device, l2_reg_linear=0, l2_reg_embedding=0, dist=args.dist)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], lr=args.lr)

    history = model.fit(train_model_input, train[target].values, batch_size=args.batch_size, epochs=args.epochs, verbose=2,
                        validation_split=0.1, args=args)
    pred_ans = model.predict(test_model_input, args.batch_size)

    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
