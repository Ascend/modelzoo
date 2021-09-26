#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
#export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=16
#网络名称，同目录名称
Network="EastV2_ID1404_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=2
#训练step
train_steps=100
#学习率
learning_rate=

#参数配置
data_path=""
ckpt_path=""

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh"
   exit 1
fi

for para in $*
do
   if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
   fi
   if [[ $para == --ckpt_path* ]];then
      ckpt_path=`echo ${para#*=}`
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

#修改参数
sed -i "s|./dataset|${data_path}|g" ${data_path}/train_data.txt
sed -i "s|./pretrained_model|${ckpt_path}|g" config.py
sed -i "s|./train_data.txt|${data_path}/train_data.txt|g" config.py
sed -i "s|max_steps = 1000|max_steps = $train_steps|g" config.py
wait

start=$(date +%s)
nohup python3 multigpu_train.py > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2e_time=$(( $end - $start ))

#参数回改
sed -i "s|${data_path}|./dataset|g" ${data_path}/train_data.txt
sed -i "s|${ckpt_path}|./pretrained_model|g" config.py
sed -i "s|${data_path}/train_data.txt|./train_data.txt|g" config.py
sed -i "s|max_steps = $train_steps|max_steps = 1000|g" config.py

#输出性能FPS，需要模型审视修改
ActualFPS=`grep "Step" $cur_path/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk -F 'examples/second' '{print $1}'|awk '{print $NF}'|awk 'END {print}'`
#打印，不需要修改

echo "Final Performance examples/sec : $ActualFPS"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN {printf "%.2f\n",'${batch_size}'*1000/'${ActualFPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "Step" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v TRACE|awk -F "total loss" '{print $2}'|awk '{print $1}' |tr -d , > $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/test/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log