#!/bin/bash

rm -rf /var/log/npu/slog/host-0/*

export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600

currentDir=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export RANK_SIZE=1

export SLOG_PRINT_TO_STDOUT=0

device_id=2

export DEVICE_ID=${device_id}
DEVICE_INDEX=${device_id}
export DEVICE_INDEX=${DEVICE_INDEX}

cd ${currentDir}/result/1p/${device_id}

python3.7 ${currentDir}/train.py --config_file vgg16_config_test > ${currentDir}/result/1p/test_${device_id}.log 2>&1

