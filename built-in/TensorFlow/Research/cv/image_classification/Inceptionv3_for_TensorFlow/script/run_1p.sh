#!/bin/bash

# set env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export HCCL_CONNECT_TIMEOUT=300

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

upDir=$(dirname "$PWD")
cd ${upDir}
work_dir=result
mkdir -p ${work_dir}

# user env
export JOB_ID=NPU20210126
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID

# debug env
export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export DUMP_GE_GRAPH=3
#export PRINT_MODEL=1

echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] train start"
python3 Incetpion_V3.py > ${work_dir}/train_0.log 2>&1
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] train end"

