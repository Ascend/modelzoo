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
#精度训练时，step设置为3000000，并做为第一个参数传入到py中，性能训练时，py内部step默认值为30000，不需要额外传入
step=3000000
start=$(date +%s)
python3 ${currentDir}/word2vec_chinese.py  $step 2>&1 | tee ${currentDir}/train.log
end=$(date +%s)
#因为word2vec中没有计算具体的精度，所以跟GPU比较的时候，我们直接看loss是否拟合，取最后十次的loss结果取平均值
#这边temp最终的值是2980000|2982000|2984000|2986000|2988000|2990000|2992000|2994000|2996000|2998000
step=$(($step-2000)) 
temp=$step
for ((i=1; i<=9; i++))
do
    step=$(($step-2000)) 
    temp="$step|$temp"
done
a=$(grep -E "$temp" ${currentDir}/train.log)
b="loss = "
c=", time cost = "
loss_result=0
function strindex() { 
  x="${1%%$2*}"
  if [[ $x = $1 ]];then
     echo -1 
  else
      echo ${#x}
      return ${#x}
  fi
}
#得到最后十次的loss值并取平均值
for ((i=1; i<=10; i++))
do
	begin_index=$(strindex "$a" "$b")
	end_index=$(strindex "$a" "$c")
	result=${a:$begin_index:$end_index-$begin_index}
	loss_result=$(echo "$loss_result+$result" | bc)
	end_index=$(($end_index+${#c}))
	a=${a:$end_index}
done
loss_result=$(echo "scale=6;$loss_result/10" | bc)
echo "Final Accuracy loss : $loss_result"
echo "Final Training Duration(s) :$(( $end - $start ))"