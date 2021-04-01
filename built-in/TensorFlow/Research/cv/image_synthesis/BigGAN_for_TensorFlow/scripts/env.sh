#!/bin/bash

rm -rf /var/log/npu/slog/host-0/*
rm -rf /var/log/npu/slog/device*


ulimit -c 0
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1

export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log
#export TF_CPP_MIN_LOG_LEVEL=2



