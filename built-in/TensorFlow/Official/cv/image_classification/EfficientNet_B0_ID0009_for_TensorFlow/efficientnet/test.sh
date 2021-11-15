#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)

# user env
export JOB_ID=9999001
export RANK_SIZE=1

export SLOG_PRINT_TO_STDOUT=0

device_id=0

currentDir=$(cd "$(dirname "$0")"; pwd)

export DEVICE_ID=${device_id}

export DEVICE_INDEX=${DEVICE_ID}

python3.7 ${currentDir}/main_npu.py \
    --data_dir=/data/slimImagenet \
    --mode=eval \
    --model_dir=result/8p/0/ \
    --eval_batch_size=128 \
    --model_name=efficientnet-b0 > eval_efficientnet-b0.log 2>&1

