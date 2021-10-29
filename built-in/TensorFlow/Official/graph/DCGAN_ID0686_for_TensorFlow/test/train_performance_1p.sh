#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="DCGAN_ID0686_for_TensorFlow"
#Batch Size
batch_size=64
#训练epoch
train_epochs=1
#学习率
learning_rate=0.0002

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_save_path           the path to save dump/profiling data, default is /home/data
    --data_path		           source data of training
    -h/--help		           show help message
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
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --data_save_path* ]];then
        data_save_path=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


#训练开始时间，不需要修改
start_time=$(date +%s)

#设置环境变量，不需要修改
echo "Device ID: $ASCEND_DEVICE_ID"
export RANK_ID=$RANK_ID

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/overflow
    over_dump_path=${cur_path}/output/$ASCEND_DEVICE_ID/overflow
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/overflow
    over_dump_path=${cur_path}/output/$ASCEND_DEVICE_ID/overflow
fi

if [ -d "/usr/local/Ascend/fwkacllib/data" ];then
    echo "data path exits."
    cp -r /usr/local/Ascend/fwkacllib/data ${cur_path}/output
else
    echo "data path is not exits."
fi

#执行训练脚本，以下传参不需要修改，其他需要模型审视修改
#--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
nohup python3.7 ../main.py \
    --dataset mnist \
    --data_dir=${data_path} \
    --input_height=28 \
    --output_height=28 \
    --epoch=${train_epochs} \
    --precision_mode=${precision_mode} \
    --train=True \
    --over_dump=${over_dump} \
    --over_dump_path=${over_dump_path} \
    --modify_mixlist=${cur_path}/../configs/ops_info.json > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#第1个迭代的耗时从统计中排除
#保存ckpt的5个迭代，每个迭代耗时大于0.05，从统计中排除
time=(`grep -E "Epoch:  0" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F, '{print $1}' | awk '{print $NF}'`)
num=0
step_sec=0
for((index=1;index<${#time[*]} - 1; index++));
do
  temp=$(awk 'BEGIN{printf "%.4f\n",('${time[index+1]}'-'${time[index]}')}')
  k=$(echo $temp 0.5 | awk '{if($1<$2) {printf "1\n"} else {printf "0\n"}}')
  if [ $k -eq 1 ] ; then
    ((num++))
    step_sec=$(awk 'BEGIN{printf "%.4f\n",('$step_sec'+'$temp')}')
  fi
done
# fps=`grep -a "FPS" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $2}'`
# performance=`grep -a "Average duration pre step" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $5}'`

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_sec=$(awk 'BEGIN{printf "%.4f\n",('$step_sec'/'$num'*'1000')}')
FPS=`awk 'BEGIN {printf "%.2f\n", '1000'*'${batch_size}'/'${step_sec}'}'`
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep -a "Final accuracy" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $4}'`
#打印，不需要修改
echo "Final train_accuracy is：$train_accuracy"
echo "E2E Training Duration sec：$e2e_time"

#看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=${step_sec}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Epoch:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F"," '{print $2}'|awk '{print $2}' > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt
#grep "Epoch:" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log |awk -F"," '{print $2}'|awk '{print $4}'|cut -c 1-10 > $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_gloss.txt

#最后一个迭代loss值
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
#ActualLoss1=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_gloss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
#echo "ActualLoss1 = ${ActualLoss1}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
