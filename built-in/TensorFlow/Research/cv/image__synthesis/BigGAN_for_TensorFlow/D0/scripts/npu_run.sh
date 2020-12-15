#!/bin/bash

#AISERVER
ulimit -c 0
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1


export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log

export RANK_SIZE=1

# inside docker
export JOB_ID=100456789
export DEVICE_ID=0
export RANK_ID=0

rm -f core.*
rm -rf ./*.pbtxt
rm -rf ./*.txt
rm -rf ./*.log
rm -rf ./kernel_meta/*
rm -rf /var/log/npu/slog/host-0/*


python3.7 BigGAN/main.py --phase train --dataset ./dataset/train/cat --epoch 10000 --iteration 100 --batch_size 64 --g_lr 0.0002 --d_lr 0.0002 --img_size 128

