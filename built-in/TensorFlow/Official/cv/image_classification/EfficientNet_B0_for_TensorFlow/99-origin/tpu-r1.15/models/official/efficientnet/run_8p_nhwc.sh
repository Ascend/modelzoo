#!/bin/bash

currentDir=$(cd "$(dirname "$0")"; pwd)

# 8p
device_id=0,1,2,3,4,5,6,7

export CUDA_VISIBLE_DEVICES=${device_id}

mkdir -p ${currentDir}/result/8p
#rm -rf ${currentDir}/result/8p/*
cd ${currentDir}/result/8p

export TF_ENABLE_AUTO_MIXED_PRECISION=1

horovodrun -np 8 -H localhost:8 python ${currentDir}/main_gpu.py --use_tpu=False \
    --data_dir=/data/slimImagenet \
    --model_dir=./ \
    --mode=train_and_eval \
    --train_batch_size=256 \
    --train_steps=218750 \
    --steps_per_eval=31250 \
    --eval_batch_size=256 \
    --transpose_input=False \
    --model_name=efficientnet-b0 > ${currentDir}/result/8p/train_8p.log 2>&1

wait

