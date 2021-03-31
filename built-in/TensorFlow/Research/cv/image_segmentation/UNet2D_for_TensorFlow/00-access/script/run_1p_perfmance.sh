#!/bin/bash


currentDir=$(cd "$(dirname "$0")"; pwd)
# set env
export PYTHONPATH=/usr/local/Ascend/ops/op_impl/built-in/ai_core/tbe/
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu:/usr/local/python3.7.5/lib/
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/toolkit/bin/:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.71.T5.0.B060
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910

export EXPERIMENTAL_DYNAMIC_PARTITION=1

export RANK_TABLE_FILE=${currentDir}/../npu_config/1p.json
ulimit -c unlimited
echo "/var/log/npu/dump/core.%e.%p" > /proc/sys/kernel/core_pattern
export DISABLE_REUSE_MEMORY=0
export TF_CPP_MIN_LOG_LEVEL=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export JOB_ID=990
export FUSION_TENSOR_SIZE=1000000000
# for producible results
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISM=1

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error

batchsize=$1

${currentDir}/exec_1p_perf.sh ${batchsize} ${DEVICE_ID}
