#!/bin/bash
CKPT=$1
DEVICE_ID=0
DEVICE_RANK=1

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
    mode freeze \
    model.input_format_dimension 4 \
    model.convert_output_to_uint8 True \
    data.eval_batch_size -1 \
    checkpoint ${CKPT}
