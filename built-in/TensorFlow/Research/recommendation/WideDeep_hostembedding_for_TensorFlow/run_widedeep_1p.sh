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
export RANK_TABLE_FILE=${currentDir}/widedeep_host1p_smoke/hccl.json

python3.7 -u ${currentDir}/widedeep_host1p_smoke/host_widedeep_1p_prec.py --data_dir=${currentDir}/tfrecord_2020_1204_threshold_100/ --max_steps=15000 --model_dir=${currentDir}/ckpt > ${currentDir}/log/train_widedeep.log 2>&1 &
echo "finish train model of widedeep"
