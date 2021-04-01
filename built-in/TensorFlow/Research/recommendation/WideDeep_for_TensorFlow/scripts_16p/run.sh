#!/bin/bash
SCRIPTPATH=$(pwd)

source /opt/npu/deepFM/miniconda3/bin/activate ci3.7
export SLOG_PRINT_TO_STDOUT=0
# export ME_DRAW_GRAPH=1
# export GE_TRAIN=1
export HCCL_FLAG=1
# export DISABLE_REUSE_MEMORY=1
# export DUMP_GE_GRAPH=2
# export DUMP_OP=1
#ge
export LD_LIBRARY_PATH=/usr/local/HiAI/runtime/lib64/:/usr/local/HiAI/driver/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/npu/NCF/software/buildtools/isl-0.16.1/lib/:$LD_LIBRARY_PATH

#ms
#export PYTHONPATH=/opt/npu/NCF/mindspore/:${PYTHONPATH}

#TBE

#export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export ME_TBE_PLUGIN_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow/
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:/usr/local/HiAI/runtime/lib64/topi.egg:/usr/local/HiAI/runtime/lib64/te.egg:${PYTHONPATH}
#/usr/local/HiAI/runtime/python3.6/site-packages/topi.egg/:/usr/local/HiAI/runtime/python3.6/site-packages/te.egg/:${PYTHONPATH}
#export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:/opt/npu/caiguohao/miniconda3/envs/yht36/lib/python3.6/site-packages:${PYTHONPATH}

#option
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so
export OPTION_PROTO_LIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so

#cce
export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:${PATH}

#gcc-7.3.0
export PATH=/opt/npu/NCF/software/buildtools/gcc-7.3.0/bin:$PATH

RANK_SIZE=$1
DEV_ID=0
RANK_ID=$2

export RANK_TABLE_FILE=${SCRIPTPATH}/tools/hccl_16p.json
export GE_TRAIN=1
echo $RANK_SIZE
echo $RANK_ID
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+0] $[DEV_ID+0] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+1] $[DEV_ID+1] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+2] $[DEV_ID+2] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+3] $[DEV_ID+3] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+4] $[DEV_ID+4] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+5] $[DEV_ID+5] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+6] $[DEV_ID+6] $RANK_SIZE
bash $SCRIPTPATH/scripts_16p/run_me_cluster.sh $[RANK_ID+7] $[DEV_ID+7] $RANK_SIZE

