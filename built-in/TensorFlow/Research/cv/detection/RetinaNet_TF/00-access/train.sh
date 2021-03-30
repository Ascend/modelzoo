#!/bin/bash

rm -rf /root/ascend/log/*
# set env
export PYTHONPATH=/usr/local/Ascend/ops/op_impl/built-in/ai_core/tbe/
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu:/usr/local/python3.7.5/lib/:/usr/local/hdf/hdf5/lib/
PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/toolkit/bin/:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.71.T5.0.B060
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
#export DUMP_GE_GRAPH=2
#export DUMP_GRAPH_LEVEL=1
#unset DUMP_GE_GRAPH
#export DUMP_GRAPH_LEVEL=1
#export PRINT_MODEL=1
#export PRINT_MODEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#export DUMP_OP=1
#export HCCL_CONNECT_TIMEOUT=6000

export EXPERIMENTAL_DYNAMIC_PARTITION=1

export RANK_TABLE_FILE=/home/swx689421/RetinaNet_TensorFlow-master/npu_config/1p.json
ulimit -c unlimited
echo "/var/log/npu/dump/core.%e.%p" > /proc/sys/kernel/core_pattern
# for fast training
#unset DUMP_OP
#unset PRINT_MODEL
#unset DUMP_GE_GRAPH
export DISABLE_REUSE_MEMORY=0
export TF_CPP_MIN_LOG_LEVEL=1
export ASCEND_GLOBAL_LOG_LEVEL=1
export RANK_SIZE=1
export RANK_ID=0
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export JOB_ID=990
export FUSION_TENSOR_SIZE=1000000000
# for producible results
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISM=1

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g info
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] train start"
python3 train.py
echo "[`date +%Y%m%d-%H:%M:%S`] [INFO] train end"

