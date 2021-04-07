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
	--device='npu' \
	--gpu=0 \
	--dist-backend 'hccl' \
	--epochs=1 \
	--batch-size=256 \
	--amp > ./resnet18_1p.log 2>&1
        