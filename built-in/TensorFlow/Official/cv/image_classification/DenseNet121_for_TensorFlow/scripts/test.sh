#!/bin/bash

# set env
export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0

dname=$(dirname "$PWD")

device_id=0

export DEVICE_ID=${device_id}
export DEVICE_INDEX=${DEVICE_ID}

#start exec
python3.7 ${dname}/train.py --rank_size=1 \
    --mode=evaluate \
    --data_dir=/opt/npu/slimImagenet \
    --eval_dir=${dname}/scripts/result/8p/0/model_8p \
    --log_dir=./ \
    --log_name=eval_densenet121.log > eval.log 2>&1

