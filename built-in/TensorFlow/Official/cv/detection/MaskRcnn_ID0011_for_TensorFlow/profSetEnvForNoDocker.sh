#!/bin/bash
#需要采集哪个device的性能数据
devId=$1
if [ -z "${devId}" ]; then
    echo "[ERROR] Please input devId"
    exit 1
fi

if [[ ! ${devId} =~ ^([0-7])$ ]]; then
    echo "[ERROR] Please input valid devId:${devId}"
    exit 1
fi
#GE Options
#按需配置为true或者false
export PROFILING_MODE=false
#按需配置，此处配置的是task粒度的性能数据+迭代轨迹的5个点信息
export PROFILING_OPTIONS=training_trace:task_trace
#按需配置，此处配置的是resnet网络的FP Start打点信息
export FP_POINT=resnet50/conv2d_1/kernel
#按需配置，此处配置的是resnet网络的BP End打点信息
export BP_POINT=gradients/box_head/fc7/Relu_grad/ReluGrad

#AICPU  Options
#按需配置为true或者false，此处配置的是不采集AICPU的数据增强的数据
export AICPU_PROFILING_MODE=false

#PROFILING  Options
inotifiDir=/var/log/npu/profiling/container/${devId}
if [ ! -d "${inotifiDir}" ]; then
    mkdir "${inotifiDir}"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Create profiling_dir:${inotifiDir} error!"
        return 1
    fi
    sleep 1
fi
export PROFILING_DIR=${inotifiDir}
return 0

