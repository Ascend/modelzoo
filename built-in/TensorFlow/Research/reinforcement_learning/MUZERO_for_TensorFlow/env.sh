export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=/usr/local/python3.7/lib/python3.7/site-packages:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp

#PYTHONPATH=${PKG_PATH}/usr/local/HiAI/lib64/:${PKG_PATH}/usr/local/HiAI/ops/op_impl/built-in/ai_core/tbe/:/usr/local/HiAI/runtime/python3.7/site-packages/te.egg:/usr/local/HiAI/runtime/python3.7/site-packages/topi.egg:/usr/local/python3/lib/python3.7/site-packages:/home/tmp_test/vega/automl/
#export PYTHONPATH=/opt/npu/resnet50_cloud_model/loos-contrust/moxing/:/home/liyong/automl/:${PYTHONPATH}:$PYTHONPATH
#export PYTHONPATH=/home/liyong/automl/:$PYTHONPATH
#export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/HiAI/runtime/lib64
#export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:$PATH
export CUSTOM_OP_LIB_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow
#export DDK_VERSION_PATH=/usr/local/HiAI/runtime/ddk_info
# export DDK_VERSION_FLAG=1.60.T17.B830
export DDK_VERSION_FLAG=1.60.T49.0.B201
#export TE_PARALLEL_COMPILER=0
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so:/usr/local/HiAI/runtime/lib64/libhccl.so
export OP_PROTOLIB_PATH=/usr/local/HiAI/runtime/ops/built-in/

#export RANK_TABLE_FILE=/home/203.json
export RANK_TABLE_FILE=/home/code/ad_conf.json

export DEVICE_ID=0
export PRINT_MODEL=1
#export DUMP_GRAPH_LEVEL=1 #xw
export DUMP_GE_GRAPH=2 # xw
#export DUMP_OP=1  # xw

#export SLOG_PRINT_TO_STDOUT=1
#export DLS_LOCAL_CACHE_PATH=${pwd}

#export RANK_TABLE_FILE=../config/1p.json
export RANK_ID=0
export RANK_SIZE=1
export JOB_ID=10087

export SOC_VERSION=Ascend910


# 
export PRINT_MODEL=1
# 

echo $PYTHONPATH


