#!/bin/bash

cur_path=`pwd`/../
#失败用例打屏
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#Batch Size
batch_size=1024
#网络名称，同目录名称
Network="MNIST_ID2481_for_TensorFlow2.X"
#Device数量，单卡默认为1
RankSize=1
#训练epoch，可选
train_epochs=10
#训练step
#train_steps=5
#学习率
#learning_rate=1e-4
#ckpt_path=""
#参数配置
data_path=""
#work_dir="$cur_path/estimator_working_dir"
#export_path="$cur_path/outputs/models/000001-first_generation"

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_performance_1p.sh $data_path --work_dir="$cur_path/estimator_working_dir" --export_path="$cur_path/outputs/models/000001-first_generation""
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

cd $cur_path

##############执行训练##########
if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

#sed -i 's/".."/$data_path/g' ./model/config.1_FRLR.unetxst.yml
#sed -i 's/"steps_per_epoch=n_batches_train"/"steps_per_epoch=5"/g' ./model/train.py
start=$(date +%s)
nohup python3 mnist_main.py --model_dir=./ckpt --train_epochs=10 --distribution_strategy=one_device --num_gpus=1 --download=False --eval_static=False --data_dir=$data_path >$cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))


#echo "Final Performance ms/step : $average_perf"
echo "Final Training Duration sec : $e2etime"

###下面字段用于冒烟看护
BatchSize=${batch_size}
#设备类型，自动获取
DeviceType=`uname -m`
#用例名称，自动获取
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取错误信息
#系统错误信息
error_msg="Graph engine process graph failed: EZ3002: Optype \[Conv2DBackpropFilter\] of Ops kernel"
#判断错误信息是否和历史状态一致，此处无需修改
Status=`grep "${error_msg}" $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
#失败阶段，枚举值图准备FAIL/图拆分FAIL/图优化FAIL/图编译FAIL/图执行FAIL/流程OK
ModelStatus="图执行FAIL"
#DTS单号或者issue链接
DTS_Number="DTS2021090622224"

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ModelStatus = ${ModelStatus}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DTS_Number = ${DTS_Number}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Status = ${Status}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "error_msg = ${error_msg}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2etime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
