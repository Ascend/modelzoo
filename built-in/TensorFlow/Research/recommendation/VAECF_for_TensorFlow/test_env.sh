#!/bin/bash

# set env
#ulimit -c unlimited

export install_path=/usr/local/Ascend

export LD_LIBRARY_PATH=/usr/local/lib/:${install_path}/fwkacllib/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:${install_path}/opp/op_impl/built-in/ai_core/tbe/op_tiling
export PYTHONPATH=${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/fwkacllib/python/site-packages:${install_path}/tfplugin/python/site-packages
export PATH=$PATH:${install_path}/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=${install_path}/opp/
export TBE_IMPL_PATH=${install_path}/opp/op_impl/built-in/ai_core/tbe

export DDK_VERSION_FLAG=1.76.22.3.220
export ASCEND_AICPU_PATH=/usr/local/Ascend
export SOC_VERSION=Ascend910

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TF_CPP_MIN_LOG_LEVEL=3 # 减少GE TF打印
export ASCEND_GLOBAL_LOG_LEVEL=3
export EXPERIMENTAL_DYNAMIC_PARTITION=1 # 动态shape

#export DUMP_GE_GRAPH=2  #生成对应算子图就与下面的这个变量一起设置为1
#export DUMP_GRAPH_LEVEL=2
#export PRINT_MODEL=1   #TF图
#export GE_USE_STATIC_MEMORY=0


