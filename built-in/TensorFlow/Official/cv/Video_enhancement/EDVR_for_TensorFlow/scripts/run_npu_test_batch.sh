#!/bin/bash
DEVICE_ID=$1
DEVICE_RANK=$2
CKPT=$3

#export PRINT_MODEL=1
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1

export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log

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

for step in $(seq 5000 5000 600000)
do
    echo "Test with ckpt of step ${step}"
    python3 tools/main.py --config-file configs/edvr.yaml \
        mode eval \
        checkpoint ${CKPT}-${step}
    echo "Test with ckpt of step ${step} done."
done

