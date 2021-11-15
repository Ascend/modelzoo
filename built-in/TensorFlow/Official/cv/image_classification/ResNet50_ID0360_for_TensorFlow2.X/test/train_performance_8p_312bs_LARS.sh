#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="ResNet50_ID0360_for_TensorFlow2.X"
#训练epoch
train_epochs=6
#训练batch_size
batch_size=2496
#训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=9.5

#TF2.X独有，需要模型审视修改
export NPU_LOOP_SIZE=${train_steps}

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_8P_312bs_LARS.sh <args>"

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
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cd $cur_path/../tensorflow

start_time=$(date +%s)
#############执行训练#########################
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID
    /usr/local/Ascend/driver/tools/msnpureport -g error -d $ASCEND_DEVICE_ID

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

    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
    nohup ${bind_core} python3 resnet_ctl_imagenet_main.py \
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
        --epochs_between_evals=1 \
        --eval_offset_epochs=2 \
        --optimizer=LARS \
        --label_smoothing=0.1 \
        --single_l2_loss_op \
        --warmup_epochs=4 \
        --weight_decay=0.0002 \
        --target_accuracy=0.9 \
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

#############结果处理#########################
echo "------------------ Final result ------------------"
step_sec=`grep TimeHistory  $cur_path/output/0/train_0.log|awk 'END {print $6}'`
echo "Final Performance ms/step : $step_sec"
step_sec=`echo ${step_sec%.*}`
e2e_sec=`expr ${train_epochs} \* 1281167 / ${step_sec} `
echo "Final Training Duration sec : $e2e_sec"
train_accuracy=`grep eval_accuracy $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v mlp_log|awk 'END {print $5}'| sed 's/,//g' |cut -c 1-5`
echo "Final train_accuracy is ${train_accuracy}"
echo "E2E training Duration sec: $e2e_time"

#############冒烟看护#########################
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量
ActualFPS=${step_sec}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RANK_SIZE}'*1000/'${step_sec}'}'`

##获取Loss
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中
grep train_loss $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v BatchTimestamp|awk '{print $10}'|sed 's/,//g'|sed '/^$/d' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中
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

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    sed -i "/AttributeError/d" $cur_path/output/${RANK_ID}/train_${RANK_ID}.log
    sed -i "/MLL/d" $cur_path/output/${RANK_ID}/train_${RANK_ID}.log 
done
