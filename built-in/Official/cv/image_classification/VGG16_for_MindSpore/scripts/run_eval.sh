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
echo "Please run the script as: "
echo "bash run_eval.sh DATA_PATH DATASET_TYPE DEVICE_TYPE CHECKPOINT_PATH"
echo "for example: bash run_eval.sh /path/ImageNet2012/train cifar10 Ascend /path/a.ckpt "
echo "=============================================================================================================="

DATA_PATH=&1
DATASET_TYPE=$2
DEVICE_TYPE=$3
CHECKPOINT_PATH=$4

python eval.py \
    --data_path=$DATA_PATH \
    --dataset=$DATASET_TYPE \
    --device_target=$DEVICE_TYPE \
    --pre_trained=$CHECKPOINT_PATH > output.eval.log 2>&1 &