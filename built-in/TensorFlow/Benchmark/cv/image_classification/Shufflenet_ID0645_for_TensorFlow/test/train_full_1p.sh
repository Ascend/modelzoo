#!/bin/bash
cur_path=`pwd`/../
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=200
#网络名称，同目录名称
Network="Shufflenet_ID0645_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=35
#训练step
train_steps=500
#学习率
learning_rate='1e-1'

#参数配置
data_path="../tiny-imagenet-200"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh --data_path=../tiny-imagenet-200"

   echo ""
   echo "parameter explain:
   --data_path        sorce data of training
   --batch_size       batchsize
   --train_epochs     train epochs
   --train_steps      train steps
   --learning_rate    A floar number, initial learning rate.
   -h/--help          Show help message
   "
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
   elif [[ $para == --train_epochs* ]];then
      train_epochs=`echo ${para#*=}`
   elif [[ $para == --train_steps* ]];then
      train_steps=`echo ${para#*=}`
   elif [[ $para == --learning_rate* ]];then
      learning_rate=`echo ${para#*=}`
   fi
done

if [[ $data_path  == "" ]];then
   echo "[Error] para \"data_path\" must be config"
   exit 1
fi
##############执行训练##########
cd $cur_path
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
nohup python3 train.py \
--train_tfrecords=${data_path}/train.tfrecords \
--val_tfrecords=${data_path}/val.tfrecords \
--batch_size=${batch_size} \
--num_epochs=${train_epochs} \
--steps_per_epoch=${train_steps} \
--initial_lr=${learning_rate} > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
steps_sec=`grep "time:"  $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END{print $10}'|awk -F, '{print $1}'`
#打印，不需要修改
FPS=`awk 'BEGIN{printf("%.2f\n",'${batch_size}'*'${train_steps}'/'${steps_sec}')}'`
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "acc:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $6}'|awk -F, '{print $1}'`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"


#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "loss:" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $2}'|awk -F, '{print $1}' >> $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log