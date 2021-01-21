#!/bin/bash
CKPT=$1
DEVICE_ID=0
DEVICE_RANK=1

#AISERVER
ulimit -c 0
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/fwkacllib/ops/framework/built-in/tensorflow/

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:${PYTHONPATH}

export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/toolkit/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
#export PRINT_MODEL=1
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

export TF_CPP_MIN_LOG_LEVEL=3

# Turn profiling on
export JOB_ID=123456789
export DEVICE_ID=${DEVICE_ID}
export DEVICE_INDEX=${DEVICE_ID}
export RANK_ID=${DEVICE_ID}
export RANK_SIZE=${DEVICE_RANK}
if [ ${DEVICE_RANK} -gt 1 ]; then
    export RANK_TABLE_FILE=scripts/${DEVICE_RANK}p.json
fi

#profiling
export PROFILING_MODE=false  # true if need profiling

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
/usr/local/Ascend/driver/tools/docker/slogd &

rm -rf aicpu*
rm -rf ge*
rm -rf *.pbtxt
rm -rf ./*.log
rm -rf kernel_meta
rm -rf /var/log/npu/slog/host-0/*


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

python3 tools/main.py \
    --config-file configs/edvr.yaml \
    mode freeze \
    checkpoint ${CKPT}
