#!/usr/bin/env bash

source npu_setenv.sh

device_id=0,1,2,3,4,5,6,7

python3.7 -u ../train.py \
    --data=/home/sfnet \
    --addr=$(hostname -I |awk '{print $1}') \
    --workers=184 \
    --print-freq=1 \
    --eval-freq=5 \
    --dist-url='tcp://127.0.0.1:50000' \
    --dist-backend='hccl' \
    --multiprocessing-distributed \
    --world-size=1 \
    --rank=0 \
    --amp \
    --loss-scale 64 \
    --batch-size 8192 \
    --learning-rate 1 \
    --wd 8e-5 \
    --device-list=${device_id} \
    --device_num 8 \
    --benchmark 0
