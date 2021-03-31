#!/bin/bash
DEVICE_ID=$1

#AISERVER
ulimit -c 0
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/fwkacllib/ops/framework/built-in/tensorflow/

#sxx 1112
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/python3/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/usr/local/Ascend/fwkacllib/ops/op_impl/built-in/ai_core/tbe/:/usr/local/Ascend/fwkacllib/python/site-packages/te:/usr/local/Ascend/fwkacllib/python/site-packages/topi:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH

export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export PRINT_MODEL=1
export MOX_USE_NPU=1
export FUSION_TENSOR_SIZE=2000000000
export MOX_USE_TF_ESTIMATOR=0
export MOX_USE_TDT=1

export HEARTBEAT=1
export CONITNUE_TRAIN=true
export LOG_DIR=./log

export DUMP_GE_GRAPH=3
export DUMP_GRAPH_LEVEL=3

export DISABLE_REUSE_MEMORY=0

# Turn profiling on
export JOB_ID=123456789
export DEVICE_ID=${DEVICE_ID}
export DEVICE_INDEX=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=1

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=1
/usr/local/Ascend/driver/tools/msnpureport -d 1 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 5 -g error

#/usr/local/Ascend/driver/tools/msnpureport -d 1 -g debug
#export ASCEND_GLOBAL_EVENT_LEVEL=0
#export ASCEND_GLOBAL_EVENT_ENABLE=0
#export ASCEND_GLOBAL_LOG_LEVEL=0


#profiling
export PROFILING_MODE=false  # true if need profiling
export PROFILING_OPTIONS="{\"output\":\"/autotest/d00564369/mmoe_ali/profiling\",\"task_trace\":\"on\",\"training_trace\":\"on\",\"aicpu\":\"on\",\"fp_point\":\"\",\"bp_point\":\"\",\"aic_metrics\":\"PipeUtilization\"}"

export OP_PROTOLIB_PATH=/usr/local/Ascend/opp/
export DDK_VERSION_FLAG=1.72.T2.0.B020

# rts fix so
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/librts_engine.so
#export DUMP_OP=0
export SOC_VERSION=Ascend910
export DDK_VERSION_PATH=/usr/local/Ascend/fwkacllib/ddk_info
export GE_AICPU_FLAG=1
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1

rm -rf aicpu*
rm -rf ge*
rm -rf *.pbtxt
rm -rf kernel_meta
rm -rf /var/log/npu/slog/host-0/*
rm -rf /var/log/npu/profiling/*

if [ $MOX_USE_TDT -eq 1 ];then
  tdt_path=/opt/npu/data/172.168.10.101/0/626363/config/tdt
  if [ ! -d ${tdt_path} ]; then
    mkdir -p ${tdt_path}
  fi
 TDTPID=$(ps -ef | grep "Tdt" | grep -v grep | awk '{print $2}')
 if [ -n "$TDTPID" ]; then
   kill -9 $TDTPID
 fi
 sleep 3
fi

time2=$(date "+%Y%m%d%H%M%S")
python3 -u ./train/main.py --config_name "mmoe_config" --tag "mmoe_transformer" --npu_mode > train_${time2}.log 2>&1 &
echo "finish train model of ${day_end}"
