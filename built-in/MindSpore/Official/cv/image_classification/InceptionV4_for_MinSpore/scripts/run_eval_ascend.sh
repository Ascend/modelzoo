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

export DEVICE_ID=$1
DATA_DIR=$2
CHECKPOINT_PATH=$3
export RANK_SIZE=1

rm -rf evaluation_ascend
mkdir ./evaluation_ascend
cd ./evaluation_ascend || exit
echo  "start training for device id $DEVICE_ID"
env > env.log
python ../eval.py --platform=Ascend --dataset_path=$DATA_DIR --checkpoint_path=$CHECKPOINT_PATH > eval.log 2>&1 &
cd ../
