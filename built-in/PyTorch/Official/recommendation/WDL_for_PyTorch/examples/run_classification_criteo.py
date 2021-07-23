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

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

parser = argparse.ArgumentParser(description='WDL')
parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training.')
parser.add_argument('--device_id', default='0', type=str, help='device id')

parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss_scale', default=128, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O1', type=str,
                    help='loss scale using in amp, default -1 means dynamic')

parser.add_argument('--data_dir', default='/data/criteo/train.txt', type=str, help='train data, and is to be')
parser.add_argument('--use_npu', default=True, type=bool, help='if use npu.')
parser.add_argument('--use_cuda', default=True, type=bool, help='if use npu.')
parser.add_argument('--distributed',action='store_true',help='Use multi-processing distributed training to launch ''N processes per node, which has N GPUs.')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--dist_rank',
                    default=0,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = parser.parse_args()
    fix_random(args.seed)

    target = ['label']
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_features + sparse_features
    data = pd.read_csv(args.data_dir, sep='\t')

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.12, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_npu = True
    if use_npu and torch.npu.is_available():
        print('npu ready...')
        device = 'npu:' + str(args.device_id)
    torch.npu.set_device(device)
    model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,dnn_activation='relu', dnn_hidden_units=(512,256),dnn_dropout=0.5,
                   task='binary',
                   l2_reg_embedding=0.0, l2_reg_linear=0.0, device=device)

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_model_input, train[target].values, batch_size=1024, epochs=2, verbose=2,
                        validation_split=0.2, args)
    pred_ans = model.predict(test_model_input, 512)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
