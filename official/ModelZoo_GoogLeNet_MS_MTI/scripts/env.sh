source /root/archiconda3/bin/activate ci3.7
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/HiAI/runtime/lib64/:/usr/local/HiAI/add-ons/
export ME_TBE_PLUGIN_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow/
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so:/usr/local/HiAI/runtime/lib64/libhccl.so
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe/
export FE_FLAG=1
export HCCL_FLAG=1
export DEVICE_ID=0
export AICPU_FLAG=1
export SLOG_PRINT_TO_STDOUT=1
export OPTION_PROTO_LIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
export PYTHONPATH=/PATH/TO/YOUR/CODE:${PYTHONPATH}