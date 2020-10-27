#!/bin/bash
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

# bash run_multigpu_train.sh RANK_SIZE EPOCH_SIZE DATASET
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
RANK_SIZE=$1
EPOCH_SIZE=$2
DATASET=$3
VOCAB_SIZE=$4
EMB_DIM=$5

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python -s ${self_path}/../train_and_eval_auto_parallel.py  \
        --device_target="GPU"                                  \
        --data_path=$DATASET                                   \
        --epochs=$EPOCH_SIZE                                   \
        --vocab_size=$VOCAB_SIZE                               \
        --emb_dim=$EMB_DIM                                     \
        --dropout_flag=1                                       \
        --host_device_mix=1 > log.txt 2>&1 &
