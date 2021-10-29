#!/bin/bash

cur_path=`pwd`/../

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#Batch Size
batch_size=32
#网络名称，同目录名称
Network="Albert_ID0632_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch，可选
train_epochs=1
#训练step
train_steps=10
#学习率
learning_rate=0.00001375

#参数配置
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path \" must be config"
   exit 1
fi
##############执行训练##########
cd $cur_path
wait

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 -m albert.run_pretraining \
              --input_file=${data_path}/data/tf_news_2016_zh_raw_news2016zh_1.tfrecord \
              --output_dir=out \
              --init_checkpoint= \
              --albert_config_file=./albert/albert_config/albert_config_base.json \
              --do_train \
              --do_eval \
              --train_batch_size=32 \
              --eval_batch_size=8 \
              --max_seq_length=512 \
              --max_predictions_per_seq=51 \
              --optimizer='lamb' \
              --learning_rate=0.00001375 \
              --num_train_steps=${train_steps} \
              --num_warmup_steps=${train_steps} \
              --save_checkpoints_steps=10 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#TrainingTime=`grep "tensorflow:examples/sec:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $2}'`
wait
#FPS=`awk 'BEGIN{printf "%.2f\n",'${batch_size}'*'${TrainingTime}'}'`
FPS=`grep "tensorflow:examples/sec:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $2}'|tail -2|head -n 1`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep ":  masked_lm_accuracy" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $4}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "tensorflow:loss =" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $3}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print $1}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log