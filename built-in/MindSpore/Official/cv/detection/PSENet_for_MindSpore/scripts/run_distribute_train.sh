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

current_exec_path=$(pwd)
echo 'current_exec_path: '${current_exec_path}

if [ $# != 1 ]
then
    echo "Usage: sh run_distribute_train.sh [PRETRAINED_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
PATH1=$(get_real_path $1)


if [ ! -f $PATH1 ]
then
    echo "error: PRETRAINED_PATH=$PATH1 is not a file"
exit 1
fi

python ${current_exec_path}/src/generate_hccn_file.py

export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE=${current_exec_path}/rank_table_4p.json

for((i=0; i<${DEVICE_NUM}; i++))
do
    if [ -d ${current_exec_path}/device_$i/ ]
    then
        if [ -d ${current_exec_path}/device_$i/checkpoints/ ]
        then
            rm ${current_exec_path}/device_$i/checkpoints/ -rf
        fi

        if [ -f ${current_exec_path}/device_$i/loss.log ]
        then
            rm ${current_exec_path}/device_$i/loss.log
        fi

        if [ -f ${current_exec_path}/device_$i/test_deep$i.log ]
        then
            rm ${current_exec_path}/device_$i/test_deep$i.log
        fi
    else
        mkdir ${current_exec_path}/device_$i
    fi

    cd ${current_exec_path}/device_$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${current_exec_path}/train.py --run_distribute --device_id $i --pre_trained $PATH1 --device_num ${DEVICE_NUM} >test_deep$i.log 2>&1 &
    cd ${current_exec_path} || exit
done

