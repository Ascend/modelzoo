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
import numpy as np
from time import time
from data_loader import load_data
from train import train
import os

# os.environ['EXPERIMENTAL_DYNAMIC_PARTITION']='1'
# os.environ['ASCEND_GLOBAL_LOG_LEVEL']='0'
# os.environ['ASCEND_GLOBAL_EVENT_ENABLE']='1'
# os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
# os.environ['ASCEND_SLOG_PRINT_TO_STDOUT']='1'
# os.environ['DUMP_GE_GRAPH']='2'
# os.environ['DUMP_GRAPH_LEVEL']='3'
print(os.getenv('PYTHONPATH'))
seed = 555  # int(time())
np.random.seed(seed)

parser = argparse.ArgumentParser()

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
'''

'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
'''

'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
'''

# restaurant
parser.add_argument('--data_path', type=str, default='../data/restaurant', help='which data_path to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')  # default 10
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--save_dir', dest='save_dir', default='../test/out/checkpoints')

show_loss = True  # default False
show_time = True
show_topk = True  # default False

t = time()

args = parser.parse_args()
data = load_data(args)
os.system('rm -rf newdata')
os.mkdir("newdata")
np.save("newdata/data.npy", data)
os.system('python3 dataset.py --n_epochs=%s --n_iter=%s --batch_size=%s --seed=%s &' % (
    args.n_epochs, args.n_iter, args.batch_size, seed))
train(args, data, show_loss, show_topk, seed)

if show_time:
    print('time used: %d s' % (time() - t))
