export RUNTIME_MODE="rpc_cloud"
export SLOG_PRINT_TO_STDOUT=1
export DEVICE_ID=0
export AICPU_FLAG=1
export FE_FLAG=1
export GLOG_v=1

export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/HiAI/runtime/lib64/:/usr/local/HiAI/driver/lib64:/usr/local/HiAI/add-ons/:${LD_LIBRARY_PATH}

export ME_TBE_PLUGIN_PATH=/usr/local/HiAI/runtime/ops/framework/built-in/tensorflow/
export PYTHONPATH=/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe:/usr/local/HiAI/runtime/lib64/topi.egg:/usr/local/HiAI/runtime/lib64/te.egg:${PYTHONPATH}

export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/libhccl.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so
export OPTION_PROTO_LIB_PATH=/usr/local/HiAI/runtime/ops/op_proto/built-in/libopsproto.so

export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:${PATH}

CURRENT_PATH=$(pwd)
#DATA PATH
export TDT_DATASET=$CURRENT_PATH/cocodataset/

#LOAD CKPT
export LOAD_CHECKPOINT_PATH=$CURRENT_PATH/train_faster_rcnn/pretrained_model/pretrain.ckpt
