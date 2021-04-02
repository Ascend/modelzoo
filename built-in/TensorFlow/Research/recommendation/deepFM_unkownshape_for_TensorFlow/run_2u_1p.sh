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
export RANK_TABLE_FILE=${currentDir}/hccl.json

python3.7 ${currentDir}/2unique_deepfm_main_record_uid_fp32.py  --data_dir=${currentDir}/general_split/ --max_epochs=15 --batch_size=16000 --max_steps=50000 > ${currentDir}/log/train_deepfm.log 2>&1 &
