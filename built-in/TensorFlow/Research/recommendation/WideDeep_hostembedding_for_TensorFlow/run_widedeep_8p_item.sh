#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)

source ${currentDir}/npu_set_env.sh

DEVICE_ID=$1

echo "start train running on device ${DEVICE_ID}"

if [ ! -d ${currentDir}/ckpt${DEVICE_ID} ]; then
  mkdir ${currentDir}/ckpt${DEVICE_ID}
fi

if [ ! -d ${currentDir}/log ]; then
  mkdir ${currentDir}/log
fi

export JOB_ID=123456789
export ASCEND_DEVICE_ID=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=8
export RANK_TABLE_FILE=${currentDir}/widedeep_host8p_smoke/hccl.json

python3.7 -u ${currentDir}/widedeep_host8p_smoke/main_run.py --data_dir=${currentDir}/criteo_tfrecord_a/ --max_steps=5000 --model_dir=${currentDir}/ckpt${DEVICE_ID} > ${currentDir}/log/train_widedeep_${DEVICE_ID}.log 2>&1 &
