#!/usr/bin/env bash

python3.7 ../main.py \
	/opt/gpu/dataset/imagenet/ \
	-a resnet18 \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed=49 \
        --workers=128 \
        --learning-rate=0.1 \
        --mom=0.9 \
        --weight-decay=1.0e-04  \
        --print-freq=1 \
        --device='gpu' \
        --gpu=2 \
        --dist-backend 'nccl' \
        --epochs=1 \
        --amp \
        --batch-size=256 > ./resnet18_1p.log 2>&1
        