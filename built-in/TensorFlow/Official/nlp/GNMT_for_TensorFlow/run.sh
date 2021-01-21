dir=`pwd`
tmp_dir=`date +%Y%m%d%H%M%S`
mkdir $tmp_dir

export JOB_ID=10086
#export PROFILING_DIR=/var/log/npu/profiling/container/0
export DEVICE_ID=2
#export PROFILING_MODE=true
#export PROFILING_OPTIONS=training_trace:task_trace
#export ENABLE_DATA_PRE_PROC=1
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=${dir}/new_rank_table_1p.json
export FUSION_TENSOR_SIZE=1000000000

export EXPERIMENTAL_DYNAMIC_PARTITION=1
# Export variables
export DUMP_GE_GRAPH=2
#export PRINT_NODE=1
#export PRINT_MODEL=1



export GE_TRAIN=0
#export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/HiAI/runtime/lib64:/usr/local/HiAI/add-ons
export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/HiAI/fwkacllib/lib64/:/usr/local/HiAI/driver/lib64/common/:/usr/local/HiAI/driver/lib64/driver/:/usr/local/HiAI/add-ons/:/usr/local/HiAI/tfplugin
#export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/HiAI/runtime/lib64:/usr/local/HiAI/add-ons/:/usr/local/HiAI/toolkit/lib64/
export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:$PATH
export SLOG_PRINT_TO_STDOUT=0
export CUSTOM_OP_LIB_PATH=/usr/local/HiAI/runtime/lib64/tbe_plugin/bert
export WHICH_OP=GEOP
export DDK_VERSION_FLAG=1.60.T17.B830
export NEW_GE_FE_ID=1
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so
export OP_PROTOLIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in
export ASCEND_OPP_PATH=/usr/local/HiAI/ops

#python nmt.py --output_dir=results
python nmt.py --output_dir=results --batch_size=1 --learning_rate=5e-4
