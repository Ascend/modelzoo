#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/python3.7.5/lib/python3.7.5/site-packages/tensorflow_core/:/usr/local/python3.7.5/lib/python3.7.5/site-packages/tensorflow_core/python/:/usr/local/Ascend/opp/:/usr/local/Ascend/acllib/lib64:/usr/local/Ascend/toolkit/lib64:/usr/local/Ascend/atc/lib64
export JOB_ID=10087
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1
#export TF_CPP_MIN_VLOG_LEVEL=0
#export GE_USE_STATIC_MEMORY=1
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin:/usr/local/Ascend/toolkit/bin:/usr/local/Ascend/atc/ccec_compiler/bin:/usr/local/Ascend/atc/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export DDK_VERSION_FLAG=1.60.T17.B830
#export NEW_GE_FE_ID=1
#export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910
export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/ops/framework/built-in/tensorflow
export PYTHONPATH=/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg:/usr/local/Ascend/atc/python/site-packages/schedule_search.egg:/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/fwkacllib/python/site-packages
export WHICH_OP=GEOP
export SOC_VERSION=Ascend910

# profiling
# export PROFILING_MODE=true
# export PROFILING_OPTIONS=task_trace:training_trace
# export AICPU_PROFILING_MODE=true
# export FP_POINT=IteratorGetNext:1
# export BP_POINT=decoder/add:0
# export PROFILING_OPTIONS=\{\"result_path\":\"/var/log/npu/profiling\"\,\"training_trace\":\"on\"\,\"task_trace\":\"on\"\,\"aicpu_trace\":\"on\"\,\"fp_point\":\"{FP_POINT}\"\,\"bp_point\":\"{BP_POINT}\"\,\"ai_core_metrics\":\"PipeUtilization\"\}

#export DISABLE_REUSE_MEMORY=0
export USE_NPU='True'
# export EXPERIMENTAL_DYNAMIC_PARTITION=1
#unset EXPERIMENTAL_DYNAMIC_PARTITION
export GEOP_KEEP=1
# export TF_CPP_MIN_VLOG_LEVEL=3


# unset PRINT_NODE
# unset SLOG_PRINT_TO_STDOUT
# unset PRINT_MODEL
# unset DUMP_GE_GRAPH


# dump graph
# export DUMP_GE_GRAPH=3
# export DUMP_GRAPH_LEVEL=2
# export PRINT_MODEL=1

rm -rf /root/ascend/log/*
rm -f *.*txt

# training parameters
max_text_len=188
max_mel_len=870
vocab_size=148
lr=1e-4
start_decay=10000
decay_steps=40000
decay_rate=0.5
epoch=1
steps=3
data_path='/home/t00495118/processed_hisi'
local_path='/home/t00495118/tacotron2/model/test1'
log_every_n_step=1


# python3 tacotron_w.py
python3 train_tactron.py --steps=$steps \
  --local_path=$local_path \
  --max_text_len=$max_text_len \
  --max_mel_len=$max_mel_len \
  --vocab_size=$vocab_size \
  --data_path=$data_path \
  --start_decay=$start_decay \
  --decay_steps=$decay_steps \
  --decay_rate=$decay_rate \
  --log_every_n_step=$log_every_n_step
