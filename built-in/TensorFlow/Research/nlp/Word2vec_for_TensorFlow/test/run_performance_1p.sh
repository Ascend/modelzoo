#!/usr/bin/env bash
currentDir=$(pwd)

export RANK_SIZE=1
export JOB_ID=99999
export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/code
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp


export DEVICE_ID=0
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
#FP_POINT="nce_loss/Mul"
#BP_POINT="gradients/nce_loss/Mul_grad/Mul_1"
#export PROFILING_MODE=true
#export PROFILING_OPTIONS=\{\"output\":\"/npu/hwx1037795/Word2vec/profiling\"\,\"training_trace\":\"on\"\,\"task_trace\":\"on\"\,\"aicpu\":\"on\"\,\"fp_point\":\"${FP_POINT}\"\,\"bp_point\":\"${BP_POINT}\"\,\"aic_metrics\":\"PipeUtilization\"\}

start=$(date +%s)
python3 ${currentDir}/word2vec_chinese.py  $step 2>&1 | tee ${currentDir}/train.log
end=$(date +%s)
echo "Final Training Duration(s) :$(( $end - $start ))"