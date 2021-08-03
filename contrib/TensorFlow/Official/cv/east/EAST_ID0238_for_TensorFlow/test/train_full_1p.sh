#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
ckpt_path=""

#基础参数，需要模型审视修改
#网络名称,同目录名称
Network="EAST_ID0238_for_TensorFlow"
#训练batch_size
batch_size=20
#训练epochs或者steps
train_steps=60000
#学习率
learning_rate=0.0001

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维测参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		         if or not over detection, default is False
    --data_dump_flag	     data dump flag, default is False
    --data_dump_step		 data dump step, default is 10
    --profiling		         if or not profiling for performance debug, default is False
    --data_save_path           the path to save dump/profiling data, default is /home/data
    --data_path		         source data of training
    -h/--help		         show help message
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
    elif [[ $para == --data_save_path* ]];then
        data_save_path=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
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
if [ -d $cur_path/output ];then
    rm -rf $cur_path/output/*
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/train_ckpt
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/eval_result
else
	mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/train_ckpt
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/eval_result
fi

python3.7 ../npu_train.py \
   --input_size=512 \
   --batch_size_per_gpu=20 \
   --checkpoint_path=./checkpoint/ \
   --text_scale=512 \
   --training_data_path=$data_path/icdar2015_2013 \
   --geometry=RBOX \
   --learning_rate=0.0001 \
   --num_readers=30 \
   --pretrained_model_path=$ckpt_path/resnet_v1_50.ckpt \
   --max_steps=60000 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1

python3.7 ../eval.py \
    --test_data_path=${data_path}/ch4_test_images \
    --checkpoint_path=${cur_path}/output/$ASCEND_DEVICE_ID/train_ckpt \
    --gpu_list="" \
    --output_dir=${cur_path}/output/$ASCEND_DEVICE_ID/eval_result > ${cur_path}/output/${ASCEND_DEVICE_ID}/eval_${ASCEND_DEVICE_ID}.log 2>&1

cd ${cur_path}/output/$ASCEND_DEVICE_ID/eval_result
zip results.zip res_img_*.txt
cd ${cur_path}/.. 
python3.7 evaluation/script.py \
    -g=./evaluation/gt.zip \
    -s=${cur_path}/output/$ASCEND_DEVICE_ID/eval_result/results.zip

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep -a "FPS" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $2}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`grep -a "avg time per step" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $5}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "total loss " $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $8}'|cut -d , -f 1 >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

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
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
