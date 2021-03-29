#!/usr/bin/env bash

python3.7 ../main.py \
	/opt/gpu/dataset/imagenet/ \
	-a resnet18 \
	--addr=$(hostname -I |awk '{print $1}') \
	--seed=49 \
	--workers=128 \
	--learning-rate=1.6 \
	--mom=0.9 \
	--weight-decay=1.0e-04  \
	--print-freq=1 \
	--dist-url='tcp://127.0.0.1:49999' \
	--multiprocessing-distributed \
	--world-size=1 \
	--rank=0 \
	--device='gpu' \
	--dist-backend 'nccl' \
	--epochs=120 \
	--amp \
	--evaluate \
	--resume ../checkpoint.pth.tar \
	--batch-size=4096 > ./resnet18_8p_eval.log 2>&1
