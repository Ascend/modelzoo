#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)
runDir=${currentDir}/recommenders-master/examples/00_quick_start

if [ ! -d ${currentDir}/ckpt ]; then
  mkdir ${currentDir}/ckpt
fi

if [ ! -d ${currentDir}/log ]; then
  mkdir ${currentDir}/log
fi

echo "train running..."

source ${currentDir}/test_env.sh

device_id=$1

export JOB_ID=10010
export ASCEND_DEVICE_ID=${device_id}
export RANK_ID=${device_id}
export RANK_SIZE=1

cd ${runDir}

python3.7 naml_MIND.py --model_path=${currentDir}/ckpt --data_path=${currentDir}/data --epochs=3 > ${currentDir}/log/train_${device_id}.log 2>&1

echo "train end, detail process log was writed in ${currentDir}/log/train_${device_id}.log"

