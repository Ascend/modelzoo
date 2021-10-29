#!/bin/bash
# 本地运行
# ./train_performance_1p.sh --data_path=/home/job_id/datasets/ShapeNet

cur_path=`pwd`/../
mkdir -p $cur_path/ckpt_gpu

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="ShapeNet_ID1138_for_TensorFlow"
#Device数量，单卡默认为1
RANK_SIZE=1
#训练epoch
train_epochs=1
#训练batch_size
batch_size=4
#训练step
train_steps=3000
#学习率
learning_rate=

# 数据集路径,保持为空,不需要修改
data_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo"usage:./train_full_1p.sh <args>"
   echo " "
   echo "parameter explain:
   --data_path		         source data of training
   -h/--help		         show help message
   "
   exit 1
fi

#参数校验，不需要修改
for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
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
nohup python3 S2_network_training.py --train_steps=${train_steps} --train_batch_size=${batch_size} --train_epochs=${train_epochs} --dataset_path=${data_path}/ShapeNet_prepro.hdf5 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait

end=$(date +%s)
e2e_time=$(( $end - $start ))

#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2e_time"

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
TrainingTime=`grep "/step" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk '{print $5}' | awk -F "ms" '{print $1}'`
wait
FPS=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'${TrainingTime}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
cat $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|tr -d '\b\r'|grep -Eo "loss: [0-9]*\.[0-9]*" |awk -F " " '{print $2}' > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "TrainAccuracy = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
