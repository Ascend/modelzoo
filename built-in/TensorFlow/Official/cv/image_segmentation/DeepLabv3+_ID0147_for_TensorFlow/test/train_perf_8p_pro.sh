#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#ENV

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/../slim

#集合通信参数,不需要修改
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/../config/88.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
ckpt_path=""
#设置默认日志级别,不需要修改
export ASCEND_SLOG_PRINT_TO_STDOUT=1

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Deeplabv3+_for_TensorFlow"
#训练epoch
train_epochs=0
#训练batch_size
batch_size=7
#训练step
train_steps=32500
#学习率
learning_rate=0.0008

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="1"
profiling=False
autotune=False

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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done
#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3 ${cur_path}/../deeplab/train_8p.py \
        --logtostderr \
        --training_number_of_steps=3000 \
        --train_split="trainval" \
        --model_variant="xception_65" \
        --atrous_rates=6 \
        --atrous_rates=12 \
        --atrous_rates=18 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --train_crop_size="513,513" \
        --train_batch_size=7 \
        --dataset="pascal_voc_seg" \
        --train_logdir=log_8p \
        --dataset_dir=${data_path}/tfrecord \
        --tf_initial_checkpoint=${ckpt_path}/xception_65_coco_pretrained/x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt \
        --fine_tune_batch_norm=False > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#1
train_accuracy="null"
#
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

###for protect
BatchSize=${batch_size}
#dev
DeviceType=`uname -m`
#case
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##errormsg
#系统错误信息
error_msg="9999: Inner Error!"
#判断错误信息是否和历史状态一致，此处无需修改
Status=`grep "${error_msg}" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
#fial stage
ModelStatus="图执行FAIL"
#DTS单号或者issue链接
DTS_Number="DTS202105150LOX4WP1H00"

#关键信息打印到CaseName.log中，此处无需修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ModelStatus = ${ModelStatus}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DTS_Number = ${DTS_Number}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Status = ${Status}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "error_msg = ${error_msg}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log