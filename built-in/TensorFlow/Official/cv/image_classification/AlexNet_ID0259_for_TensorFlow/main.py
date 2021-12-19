#
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
#
import data_preprocessor
import train
import os
import gif_maker
import argparse
###########################################################
# differences with reference paper(ImageNet Classification with Deep Convolutional Neural Networks) :
# our dataset is 'cat vs dog'
# our initial learning rate(hyper parameter) is 1/10 of reference paper's
###########################################################


#if not(os.path.exists('data/data_preprocessed')):
#    DP = data_preprocessor.DataPreprocessor()
#    DP.run()

input_size = 100
lr = 0.001    # 1/10 of reference paper's initial learning rate
momentum = 0.9
decaying_factor = 0.0005
LRN_depth = 5
LRN_bias = 2
LRN_alpha = 0.0001
LRN_beta = 0.75
keep_prob = 0.5
#
from hccl.split.api import set_split_strategy_by_size
loss_sampling_step = 20
acc_sampling_step = 1

parse = argparse.ArgumentParser()
parse.add_argument("--data_path", metavar='DIR', default = "",help="path to dataset.")
parse.add_argument("--step", type=int, default=0, help="max step.")
parse.add_argument("--mul_device_id", type=int, default=0, help="device id.")
parse.add_argument("--mul_rank_size", type=int, default=1, help="number of rank size")
parse.add_argument("--epoch", type=int, default=1, help="max epoch.")
args = parse.parse_args()
max_step = args.step
max_epoch = args.epoch
mul_rank_size = args.mul_rank_size
mul_device_id = args.mul_device_id
data_path = os.path.join(args.data_path,"data/data_preprocessed/train")
#
if mul_rank_size != 1:
    lr = lr * mul_rank_size
    set_split_strategy_by_size([60, 19, 21]) 
alexnet = train.AlexNet(input_size, lr, momentum, decaying_factor, LRN_depth, LRN_bias, LRN_alpha, LRN_beta, keep_prob)
alexnet.run(max_epoch, loss_sampling_step, acc_sampling_step, max_step,data_path,mul_rank_size,mul_device_id)
alexnet.save_acc()
alexnet.save_loss()
#gif_maker.run(max_epoch)
