export SUMO_HOME="/home/sumo-master"
export PATH="/home/sumo-master/bin:$PATH"

export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/python3/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/usr/local/python3.7/lib/python3.7/site-packages:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/fwkacllib/ops/op_impl/built-in/ai_core/tbe/
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/fwkacllib/ops/framework/built-in/tensorflow/


export DDK_VERSION_FLAG=1.72.T2.0.B020

export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/librts_engine.so
export OP_PROTOLIB_PATH=/usr/local/Ascend/opp/

#export RANK_TABLE_FILE=/home/203.json

export DEVICE_ID=0
export DEVICE_INDEX=0
export PRINT_MODEL=1
export DUMP_GRAPH_LEVEL=1
export DUMP_GE_GRAPH=1


export RANK_TABLE_FILE=/home/RL2D/hccl.json
export RANK_ID=0
export RANK_SIZE=1
export RANK_INDEX=0
export JOB_ID=10087

export SOC_VERSION=Ascend910

echo $PYTHONPATH

#bash rm_logs.sh

#python3 tf_train.py

