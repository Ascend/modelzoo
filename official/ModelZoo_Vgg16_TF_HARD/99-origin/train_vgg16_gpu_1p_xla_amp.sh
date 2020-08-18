#!/bin/bash

if [ ! -d "result/1p" ]; then
    mkdir -p result/1p
else
    rm -rf result/1p/*
fi

currentDir=$(cd "$(dirname "$0")"; pwd)

# 1p
device_id=7

export CUDA_VISIBLE_DEVICES=${device_id}

mkdir -p ${currentDir}/result/1p
rm -rf ${currentDir}/result/1p/*
cd ${currentDir}/result/1p

export TF_ENABLE_AUTO_MIXED_PRECISION=1

horovodrun -np 1 -H localhost:1 python ${currentDir}/train.py --config_file vgg16_config_1p_gpu > ${currentDir}/result/1p/train_1p.log 2>&1

wait

