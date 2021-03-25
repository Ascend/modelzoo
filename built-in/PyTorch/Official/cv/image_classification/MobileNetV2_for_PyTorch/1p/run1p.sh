#!/usr/bin/env bash

source npu_set_env.sh

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"


python3.7 ${currentDir}/main_apex.py \
    --seed 49  \
    --workers 128 \
    --lr 0.05 \
    --amp \
    --opt-level 'O2' \
    --loss-scale-value 64 \
    --momentum 0.9 \
    --batch-size 512 \
    --weight-decay 1e-5 \
    --epochs 600 \
    --print-freq 1 \
    --eval-freq 1 \
    --device 'npu:0' \
    --summary-path './runs/mobilenetv2/log' \
    --data /data/imagenet > ${train_log_dir}/train_1p.log 2>&1 &
