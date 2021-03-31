#!/bin/bash

# main env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/code/models:/usr/local/python3.7/lib/python3.7/site-packages/npu_bridge/helper/
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID=job9999001
export RANK_TABLE_FILE=/d_solution/config/hccl.json
export RANK_SIZE=1
export RANK_INDEX=0
export DEVICE_INDEX=0
export DEVICE_ID=0
export RANK_ID=cloud-localhost-20200528103242-0

# profiling env
#export TF_CPP_MIN_VLOG_LEVEL=1
export PROFILING_MODE=false
export AICPU_PROFILING_MODE=false
export PROFILING_OPTIONS=training_trace:task_trace
export FP_POINT=bert/embeddings/GatherV2
export BP_POINT=gradients/bert/embeddings/MatMul_grad/MatMul_1

python3.7 main.py --exec_mode=train --batch_size=1 --max_steps=10  --data_dir=../../data/ --model_dir=ckpt0
