#!/bin/bash

DEVICE_ID=$1
DEVICE_RANK=$2
DATASET=$3
MODEL=$4

#AISERVER
ulimit -c 0

#export PRINT_MODEL=1
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1

export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log


# Turn profiling on
export JOB_ID=123456789
export DEVICE_ID=${DEVICE_ID}
export DEVICE_INDEX=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=${DEVICE_RANK}
if [ ${DEVICE_RANK} -gt 1 ]; then
    export RANK_TABLE_FILE=scripts/${DEVICE_RANK}p.json
fi


export SOC_VERSION=Ascend910
/usr/local/Ascend/driver/tools/docker/slogd &

rm -rf aicpu*
rm -rf ge*
rm -rf *.pbtxt
rm -rf ./*.log
rm -rf kernel_meta
#rm -rf /var/log/npu/slog/host-0/*

python3 graphsage/supervised_train.py --train_prefix data/${DATASET}/${DATASET} --model ${MODEL} --base_log_dir outputs/${DATASET} --device npu --device_ids ${DEVICE_ID} --rank_size ${DEVICE_RANK}
