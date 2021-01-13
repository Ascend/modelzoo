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

python3.7 ${currentDir}/densenet121_1p_main.py \
        --workers 40 \
        --arch densenet121 \
        --npu 0 \
        --lr 0.1 \
        --momentum 0.9 \
        --amp \
        --print-freq 1 \
        --eval-freq 5 \
        --batch-size 256 \
        --epoch 90 \
        --data /data/imagenet/ > ./densenet121_1p.log 2>&1 &