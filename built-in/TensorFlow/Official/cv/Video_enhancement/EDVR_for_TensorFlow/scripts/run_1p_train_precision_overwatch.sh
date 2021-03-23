#!/bin/bash
DEVICE_ID=$1
DEVICE_RANK=$2

# set env

#export PRINT_MODEL=1
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1

export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log

export ASCEND_GLOBAL_EVENT_LEVEL=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TF_CPP_MIN_LOG_LEVEL=3

# Turn profiling on
export JOB_ID=123456789
export DEVICE_ID=${DEVICE_ID}
export DEVICE_INDEX=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=${DEVICE_RANK}
if [ ${DEVICE_RANK} -gt 1 ]; then
    export RANK_TABLE_FILE=scripts/${DEVICE_RANK}p.json
fi

rm -rf kernel_meta

python3 tools/main.py \
    --config-file configs/edvr_precision_overwatch.yaml \
    solver.checkpoint_interval 5000 \
    solver.print_interval 20 \
    rank_size ${DEVICE_RANK}
