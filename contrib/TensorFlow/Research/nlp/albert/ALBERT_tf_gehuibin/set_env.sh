#!/bin/bash
# set env
export ASCEND_HOME=/usr/local/Ascend
export Ascend_toolkit=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux
#dirver包依赖
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:${Ascend_toolkit}/fwkacllib/lib64/:${ASCEND_HOME}/driver/lib64/common/:${ASCEND_HOME}/driver/lib64/driver/:${ASCEND_HOME}/add-ons
#fwkacllib包依赖
export PYTHONPATH=$PYTHONPATH:${Ascend_toolkit}/opp/op_impl/built-in/ai_core/tbe
export PYTHONPATH=$PYTHONPATH:${Ascend_toolkit}/fwkacllib/python/site-packages
#ftplugin包依赖
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages
export PATH=$PATH:${Ascend_toolkit}/fwkacllib/ccec_compiler/bin:/usr/local/python3.7/bin
export ASCEND_OPP_PATH=${Ascend_toolkit}/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
# user env
export JOB_ID=9999001
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0