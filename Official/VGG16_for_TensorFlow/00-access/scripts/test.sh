#!/bin/bash

# set env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID=9999001
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0

dname=$(dirname "$PWD")

device_id=2

export DEVICE_ID=${device_id}
export DEVICE_INDEX=${DEVICE_ID}

#start exec
python3.7 ${dname}/train.py --rank_size=1 \
    --mode=evaluate \
    --data_dir=/opt/npu/slimImagenet \
    --eval_dir=${dname}/scripts/result/8p/2/model_8p \
    --log_dir=./ \
    --log_name=eval_vgg16.log > eval.log 2>&1

