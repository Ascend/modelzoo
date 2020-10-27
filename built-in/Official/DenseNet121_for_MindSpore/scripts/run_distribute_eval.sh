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
echo "sh scripts/run_distribute_eval.sh DEVICE_NUM RANK_TABLE_FILE DATASET CKPT_PATH"
echo "for example: sh scripts/run_distribute_train.sh 8 /data/hccl.json /path/to/dataset /path/to/ckpt"
echo "It is better to use absolute path."
echo "================================================================================================================="

echo "After running the scipt, the network runs in the background. The log will be generated in eval_x/log.txt"

export RANK_SIZE=$1
export RANK_TABLE_FILE=$2
DATASET=$3
CKPT_PATH=$4

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    rm -rf eval_$i
    mkdir ./eval_$i
    cp ./*.py ./eval_$i
    cp -r ./src ./eval_$i
    cd ./eval_$i || exit
    export RANK_ID=$i
    echo "start infering for rank $i, device $DEVICE_ID"
    env > env.log
    python eval.py  \
    --data_dir=$DATASET  \
    --pretrained=$CKPT_PATH > log.txt 2>&1 &

    cd ../
done

