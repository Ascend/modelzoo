#!/usr/bin/env bash
source scripts/set_npu_env.sh 
python3.7 ./main.py \
	/opt/npu/imagenet/ \
	-a resnet18 \
	--addr=$(hostname -I |awk '{print $1}') \
	--seed=49 \
	--workers=$(nproc) \
	--learning-rate=1.6 \
	--mom=0.9 \
	--weight-decay=1.0e-04  \
	--print-freq=1 \
	--dist-url='tcp://127.0.0.1:41111' \
	--dist-backend 'hccl' \
	--multiprocessing-distributed \
	--world-size=1 \
	--rank=0 \
	--device='npu' \
	--epochs=120 \
	--amp \
	--batch-size=4096 > ./resnet18_8p.log 2>&1