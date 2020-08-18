#!/bin/bash

if [ ! -d "result/8p" ]; then
    mkdir -p result/8p
else
    rm -rf result/8p/*
fi

currentDir=$(cd "$(dirname "$0")"; pwd)

# 8p
device_id=0,1,2,3,4,5,6,7

export CUDA_VISIBLE_DEVICES=${device_id}

mkdir -p ${currentDir}/result/8p
rm -rf ${currentDir}/result/8p/*
cd ${currentDir}/result/8p

export TF_ENABLE_AUTO_MIXED_PRECISION=1

horovodrun -np 8 -H localhost:8 python ${currentDir}/train.py --config_file vgg16_config_8p_gpu > ${currentDir}/result/8p/train_8p.log 2>&1

wait

