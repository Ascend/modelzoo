#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=16
export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_16p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#####基础参数，需要模型审视修改
#Batch Size
batch_size=4096
#网络名称
Network="ResNet50_ID0360_for_TensorFlow2.X"
#Device数量
RankSize=16
#训练epoch
train_epochs=42
# 训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=7.92

#TF2.X独有，需要模型审视修改
export NPU_LOOP_SIZE=${train_steps}

#npu param
precision_mode="allow_mix_precision"
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
data_path=""

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_16P_256bs_SGD_0.sh <args>"

    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --data_path		           source data of training
    -h/--help		           show help message
    "
    exit 1
fi

#参数校验，需要模型审视修改
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
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

cd $cur_path/../tensorflow
#############执行训练#########################
for((RANK_ID=$RANK_ID_START;RANK_ID<$((8+RANK_ID_START));RANK_ID++));
do
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi
    
    #绑核，不需要绑核的模型删除，需要绑核的模型根据实际修改
    cpucount=`lscpu | grep "CPU(s):" | head -n 1 | awk '{print $2}'`
    cpustep=`expr $cpucount / 8`
    echo "taskset c steps:" $cpustep
    let a=RANK_ID*$cpustep
    let b=RANK_ID+1
    let c=b*$cpustep-1

    nohup taskset -c $a-$c python3 resnet_ctl_imagenet_main.py \
        --data_dir=${data_path} \
        --train_steps=${train_steps} \
        --num_accumulation_steps=1 \
        --train_epochs=${train_epochs} \
        --model_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt \
        --distribution_strategy=off \
        --use_tf_while_loop=true \
        --use_tf_function=true \
        --enable_checkpoint_and_export \
        --steps_per_loop=${train_steps} \
        --base_learning_rate=${learning_rate} \
        --momentum=0.901 --epochs_between_evals=1 \
        --eval_offset_epochs=2 \
        --optimizer=SGD \
        --label_smoothing=0.1 \
        --single_l2_loss_op \
        --warmup_epochs=5 \
        --weight_decay=0.000025 \
        --lr_schedule=polynomial \
        --drop_eval_remainder=True \
        --precision_mode=${precision_mode} \
        --over_dump=${over_dump} \
        --over_dump_path=${over_dump_path} \
        --data_dump_flag=${data_dump_flag} \
        --data_dump_step=${data_dump_step} \
        --data_dump_path=${data_dump_path} \
        --batch_size=${batch_size} \
        --profiling=${profiling} \
        --profiling_dump_path=${profiling_dump_path} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep TimeHistory  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $6}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep eval_accuracy $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v mlp_log|awk 'END {print $5}'| sed 's/,//g' |cut -c 1-5`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RANK_SIZE}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep train_loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v BatchTimestamp|awk '{print $10}'|sed 's/,//g'|sed '/^$/d' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
