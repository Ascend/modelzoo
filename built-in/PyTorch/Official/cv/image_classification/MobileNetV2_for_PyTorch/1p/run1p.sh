#!/usr/bin/env bash

source npu_set_env.sh
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 172.17.0.1:22118 --log \"SetLogLevel(0)[error]\" --device 0"

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
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
