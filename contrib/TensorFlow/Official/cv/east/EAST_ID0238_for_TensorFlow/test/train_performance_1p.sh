
#!/bin/bash

cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3


#基础参数，需要模型审视修改
#Batch Size
batch_size=64
#网络名称,同目录名
Network="EAST_ID0238_for_TensorFlow"
#Device数量，单卡默认为1
RankSize=1
#训练epoch
train_epochs=1
#学习率
learning_rate=0.0002

#npu param
precision_mode="allow_mix_precision"
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
data_save_path="/home/data"
data_path=""
ckpt_path=""

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1P.sh <args>"

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
    elif [[ $para == --ckpt_path* ]];then
        ckpt_path=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

export JOB_ID=10087
export ASCEND_GLOBAL_LOG_LEVEL=3
/usr/local/Ascend/driver/tools/msnpureport -g error -d ${ASCEND_DEVICE_ID}
export NPU_LOOP_SIZE=$train_steps


if [ ! -d "${cur_path}/output/${ASCEND_DEVICE_ID}" ];then
	mkdir -p ${cur_path}/output/${ASCEND_DEVICE_ID}
elif [ -d "${cur_path}/output/ckpt_new" ];then
    rm -rf ${cur_path}/output/ckpt_new
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
--max_steps=400 > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1

avg_time_per_step=`grep -a "avg time per step" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $5}'`
FPS=`grep -a "FPS" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $2}'`
accuracy=`grep -a "Final Train Accuracy" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $4}'`
e2e_time=`grep -a "E2E Training Duration sec" ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F" " '{print $5}'`

echo "Final Performance images/sec : $FPS"
echo "Final Train Accuracy : ${accuracy}"
echo "E2E Training Duration sec : $e2e_time"


#############冒烟看护#########################
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取性能数据
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=${avg_time_per_step}

#最后一个迭代loss值
ActualLoss=${accuracy}

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
