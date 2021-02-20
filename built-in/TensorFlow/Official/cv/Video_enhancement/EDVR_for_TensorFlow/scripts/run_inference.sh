#!/bin/bash
DEVICE_ID=$1
CKPT=$2
DEVICE_RANK=1

#AISERVER
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
    --config-file configs/edvr.yaml \
    mode inference \
    data.data_dir 'data/reds' \
    data.eval_in_size 180,320 \
    model.convert_output_to_uint8 True \
    checkpoint ${CKPT}
