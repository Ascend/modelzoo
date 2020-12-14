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
echo "sh run_standalone_train_ascend.sh DATASET_SCHEMA_TRAIN PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_standalone_train_ascend.sh \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.json \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.tfrecord-001-of-001"
echo "It is better to use absolute path."
echo "=============================================================================================================="

DATASET_SCHEMA_TRAIN=$1
PRE_TRAIN_DATASET=$2

export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1
export GLOG_v=2
current_exec_path=$(pwd)
echo ${current_exec_path}
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train
cp -r ../config ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py \
	--config=${current_exec_path}/train/config/config.json \
	--dataset_schema_train=$DATASET_SCHEMA_TRAIN \
	--pre_train_dataset=$PRE_TRAIN_DATASET > log_gnmt_network${i}.log 2>&1 &
cd ..
