#!/usr/bin/env bash
source pt.sh
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1

device_id_list=0,1,2,3,4,5,6,7

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

python3.7 ${currentDir}/main.py \
    --addr=$(hostname -I |awk '{print $1}') \
    --seed=49 \
    --workers=184 \
    --learning-rate=0.01 \
    --print-freq=1 \
    --eval-freq=1 \
    --dist-url 'tcp://127.0.0.1:50002' \
    --multiprocessing-distributed \
    --world-size 1 \
    --batch-size 256 \
    --device 'npu' \
    --epochs 1 \
    --rank 0 \
    --device-list '1' \
    --amp \
    --opt-level 'O1' \
    --dist-backend 'hccl' \
    --loss-scale-value 8 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --data=/opt/npu/dataset/imagenet >output_1p.log
