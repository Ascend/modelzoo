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

echo "=============================================================================================================="
echo "Please run the scipt as: "
echo "sh run_distributed_train_ascend.sh RANK_TABLE_ADDR DATASET_SCHEMA_TRAIN PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_distributed_train_ascend.sh \
  /home/workspace/rank_table_8p.json \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.json \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.tfrecord-001-of-001"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_TABLE_ADDR=$1
DATASET_SCHEMA_TRAIN=$2
PRE_TRAIN_DATASET=$3

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_ADDR
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_ADDR

echo $RANK_TABLE_FILE
export RANK_SIZE=8
export GLOG_v=2

for((i=0;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py .
    cp -r ../../src .
    cp -r ../../config .
    export RANK_ID=$i
    export DEVICE_ID=$i
	python ../../train.py \
	  --config=${current_exec_path}/device${i}/config/config.json \
	  --dataset_schema_train=$DATASET_SCHEMA_TRAIN \
	  --pre_train_dataset=$PRE_TRAIN_DATASET > log_gnmt_network${i}.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
