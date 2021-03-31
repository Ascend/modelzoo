#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)

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

python3.7 ${currentDir}/VAE-CF/main.py --data_dir=${currentDir}/data/movielens_data/ --epochs=1 --train --checkpoint_dir=${currentDir}/ckpt > ${currentDir}/log/train_${device_id}.log 2>&1

echo "train end, detail process log was writed in ${currentDir}/log/train_${device_id}.log"

