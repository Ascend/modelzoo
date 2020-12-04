#!/usr/bin/env bash
source npu_set_env.sh
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
python3.7 ${currentDir}/DistributedResnet50/main-apex-d76-npu.py \
        --data /data/imagenet \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=128 \
        --learning-rate=1.6 \
        --warmup=8 \
        --label-smoothing=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --static-loss-scale=128 \
        --print-freq=1 \
        --dist-url='tcp://127.0.0.1:50000' \
        --dist-backend='hccl' \
        --multiprocessing-distributed \
        --world-size=1 \
        --rank=0 \
        --benchmark=0 \
        --device='npu' \
        --epochs=90 \
        --batch-size=4096 > ./resnet50_8p.log 2>&1 &


