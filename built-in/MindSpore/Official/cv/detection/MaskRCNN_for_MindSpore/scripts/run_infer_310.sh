#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [[ $# -lt 3 || $# -gt 4 ]]; then 
    echo "Usage: sh run_infer_310.sh [AIR_PATH] [DATA_PATH] [ANN_FILE_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be setted by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
ann_file=$(get_real_path $3)
if [ $# == 4 ]; then
    device_id=$4
elif [ $# == 3 ]; then
    if [ -z $device_id ]; then
        device_id=0
    else
        device_id=$device_id
    fi
fi

echo $model
echo $data_path
echo $ann_file
echo $device_id

export ASCEND_HOME=/usr/local/Ascend/
export PATH=$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ones:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/atc/python/site-packages/te.egg:$ASCEND_HOME/atc/python/site-packages/topi.egg:$ASCEND_HOME/atc/python/site-packages/auto_tune.egg::$ASCEND_HOME/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export ASCEND_OPP_PATH=$ASCEND_HOME/opp

function air_to_om()
{
    atc --input_format=NCHW --framework=1 --model=$model --input_shape="x:1, 3, 768, 1280; im_info: 1, 4" --output=maskrcnn --insert_op_conf=../src/aipp.cfg --precision_mode=allow_fp32_to_fp16 --soc_version=Ascend310 &>atc.log
}

function compile_app()
{
    cd ../ascend310_infer/src
    sh build.sh &> build.log
    cd -
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
     if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ../ascend310_infer/src/out/main --om_path=./maskrcnn.om --data_path=$data_path --device_id=$device_id &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --ann_file=$ann_file --img_path=$data_path &> acc.log &
}

air_to_om
if [ $? -ne 0 ]; then
    echo "air to om failed"
    exit 1
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo "excute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi