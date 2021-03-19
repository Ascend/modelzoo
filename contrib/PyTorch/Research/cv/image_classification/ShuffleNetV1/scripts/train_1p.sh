#!/usr/bin/env bash

source npu_setenv.sh

device_id=0

python3.7 -u ../train.py \
    --data=/home/sfnet \
    --addr=$(hostname -I |awk '{print $1}') \
    --workers=8 \
    --print-freq=1 \
    --eval-freq=5 \
    --dist-url='tcp://127.0.0.1:50000' \
    --dist-backend='hccl' \
    --multiprocessing-distributed \
    --world-size=1 \
    --rank=0 \
    --amp \
    --loss-scale 64 \
    --device-list=${device_id} \
    --benchmark 0
