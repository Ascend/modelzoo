#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)

if [ ! -d ${currentDir}/ckpt ]; then
  mkdir ${currentDir}/ckpt
fi

if [ ! -d ${currentDir}/log ]; then
  mkdir ${currentDir}/log
fi

echo "train running..."

source ${currentDir}/npu_set_env.sh

DEVICE_ID=$1

export JOB_ID=123456789
export ASCEND_DEVICE_ID=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=1

python3.7 -u ${currentDir}/train/main.py --config_name "mmoe_config" --tag "mmoe_transformer" --npu_mode > ${currentDir}/log/train_MMOE.log 2>&1 &
echo "finish train model of MMOE"
