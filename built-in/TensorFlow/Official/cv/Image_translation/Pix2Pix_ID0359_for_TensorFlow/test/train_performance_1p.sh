#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
# export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network=Pix2Pix_ID0359_for_TensorFlow
#训练epoch
train_epochs=2
#训练batch_size
batch_size=1

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1P.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump              if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step         data dump step, default is 10
    --profiling              if or not profiling for performance debug, default is False
    --data_path              source data of training
    -h/--help                show help message
    "
    exit 1
fi

#参数校验，不需要修改
for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
        over_dump_path=${cur_path}/output/overflow_dump
        mkdir -p ${over_dump_path}
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
        data_dump_path=${cur_path}/output/data_dump
        mkdir -p ${data_dump_path}
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
        profiling_dump_path=${cur_path}/output/profiling
        mkdir -p ${profiling_dump_path}
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --epoch* ]];then
        train_epochs=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be config"
    exit 1
fi


#############执行训练#########################
#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 ${cur_path}/../main.py \
      --phase=train \
      --epoch=${train_epochs} \
      --data_path=${data_path} \
      --checkpoint_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt \
      --precision_mode=${precision_mode} \
      --over_dump=${over_dump} \
      --over_dump_path=${over_dump_path} \
      --data_dump_flag=${data_dump_flag} \
      --data_dump_step=${data_dump_step} \
      --data_dump_path=${data_dump_path} \
      --batch_size=${batch_size} \
      --blacklist_path=${cur_path}/../ \
      --profiling=${profiling} \
      --profiling_dump_path=${profiling_dump_path}  > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
if [ $? -ne 0 ];then
  exit 1
fi
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))
#############结果处理#########################
#跑两个Epoch后去掉checkpoint保存耗时（耗时大约9s左右），取单步耗时的平均值为最终的耗时
time=(`grep -E "Epoch: \[ 0\]|Epoch: \[ 1\]" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F, '{print $1}' | awk '{print $NF}' | cut -d ',' -f 1`)
num=0
step_sec=0
for((index=1;index<${#time[*]} - 1; index++));
do
  temp=$(awk 'BEGIN{printf "%.4f\n",('${time[index+1]}'-'${time[index]}')}')
  k=$(echo $temp 1 | awk '{if($1<$2) {printf "1\n"} else {printf "0\n"}}')
  if [ $k -eq 1 ] ; then
    ((num++))
    step_sec=$(awk 'BEGIN{printf "%.4f\n",('$step_sec'+'$temp')}')
  fi
done
#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_sec=$(awk 'BEGIN{printf "%.4f\n",('$step_sec'/'$num'*'1000')}')
#打印，不需要修改
FPS=`awk 'BEGIN {printf "%.2f\n", '1000'*'${batch_size}'/'${step_sec}'}'`
echo "Final Performance images/sec : $FPS"
#输出训练精度,需要模型审视修改
train_accuracy=''
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=${step_sec}

grep "Epoch:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F, '{print $2}' | awk '{print $NF}' | cut -d ',' -f 1 > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_dloss.log
grep "Epoch:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F, '{print $3}' | awk '{print $2}' | sed 's/W....//' > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_gloss.log
dloss=(`tail -10 $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_dloss.log`)
gloss=(`tail -10 $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}_gloss.log`)
dloss_result=0
gloss_result=0
for((index=0;index<10; index++));
do
    dloss_result=$(awk 'BEGIN{printf "%.4f\n",('${dloss_result}'+'${dloss[index]}')}')
    gloss_result=$(awk 'BEGIN{printf "%.4f\n",('${gloss_result}'+'${gloss[index]}')}')
done
dloss_result=$(awk 'BEGIN{printf "%.4f\n",('${dloss_result}'/'10')}')
gloss_result=$(awk 'BEGIN{printf "%.4f\n",('${gloss_result}'/'10')}')

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${batch_size}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${dloss_result}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss1 = ${gloss_result}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime= ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
