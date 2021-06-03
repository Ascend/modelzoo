#!/bin/bash

export NPU_EXPERIMENTAL_AUTO_LOOP=true
export ASCEND_GLOBAL_LOG_LEVEL=3
export NPU_ENABLE_PERF=true

#base param
#Batch Size
batch_size=4096
#网络名称
Network="ResNet50_TF2.X_for_TensorFlow"
#Device数量
RankSize=16
#训练epoch
train_epochs=42
# 训练step
train_steps=`expr 1281167 / ${batch_size}`
#学习率
learning_rate=9.5
export NPU_LOOP_SIZE=${train_steps}

#npu param
precision_mode="allow_mix_precision"
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
data_save_path="/home/data"
data_path=""

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_16P_256bs_LARS_173.sh <args>"

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
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

loop_size=${NPU_LOOP_SIZE}

BASE=`pwd`
MODEL_DIR=${BASE}/../tensorflow
TABLE_DIR=${BASE}/../configs

export RANK_SIZE=16
export RANK_TABLE_FILE=${TABLE_DIR}/rank_table_16p.json

cd $MODEL_DIR

find -name "ge_*" -print0 | xargs -i -0 rm -f {}
find -name "*.pbtxt" -print0 | xargs -i -0 rm -f {}

#############执行训练#########################
for i in 0 1 2 3 4 5 6 7 
do
    if [ ! -d "${BASE}/output/${i}" ];then
	    mkdir -p ${BASE}/output/${i}
    elif [ -d "${BASE}/output/ckpt_bs256_LARS_16P_${i}" ];then
        rm -rf ${BASE}/output/ckpt_bs256_LARS_16P_${i}
    fi

    export RANK_ID=`expr $i + 8`
    export ASCEND_DEVICE_ID=$i
    /usr/local/Ascend/driver/tools/msnpureport -g error -d $i
    let a=i*12
    let b=i+1
    let c=b*12-1

    nohup taskset -c $a-$c python3 resnet_ctl_imagenet_main.py --data_dir=${data_path} --train_steps=${train_steps} --num_accumulation_steps=1 --train_epochs=${train_epochs} -model_dir=${BASE}/output/ckpt_bs256_LARS_16P_$i --distribution_strategy=off --use_tf_while_loop=true --use_tf_function=true --enable_checkpoint_and_export --steps_per_loop=${loop_size} --base_learning_rate=${learning_rate} --epochs_between_evals=1 --eval_offset_epochs=2 --optimizer=LARS --label_smoothing=0.1 --single_l2_loss_op --warmup_epochs=4 --weight_decay=0.0002 --lr_schedule=polynomial --drop_eval_remainder=True --precision_mode=${precision_mode} --over_dump=${over_dump} --data_dump_flag=${data_dump_flag} --data_dump_step=${data_dump_step} --batch_size=${batch_size} --profiling=${profiling} --data_save_path=${data_save_path} > ${BASE}/output/${i}/train_${i}.log 2>&1 &
done
wait

#############结果处理#########################
echo "------------------ Final result ------------------"
step_sec=`grep TimeHistory  $BASE/output/0/train_0.log|awk 'END {print $6}'`
echo "Final Performance ms/step : $step_sec"
step_sec=`echo ${step_sec%.*}`
e2e_sec=`expr ${train_epochs} \* 1281167 / ${step_sec} `
echo "Final Training Duration sec : $e2e_sec"
train_accuracy=`grep train_accuracy $BASE/output/0/train_0.log|awk 'END {print $8}'|cut -c 1-5`
echo "Final train_accuracy is ${train_accuracy}"


#############冒烟看护#########################
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}_bs${BatchSize}_${RankSize}'p'_'perf'

##获取性能数据
#吞吐量
ActualFPS=${step_sec}
#单迭代训练时长
TrainingTime=`expr ${batch_size} \* 1000 / ${step_sec}`

##获取Loss
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中
grep train_loss $BASE/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|grep -v BatchTimestamp|awk '{print $10}'|sed 's/,//g'|sed '/^$/d' >> $BASE/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值
ActualLoss=`awk 'END {print}' $BASE/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" > $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RankSize}" >> $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $BASE/output/$ASCEND_DEVICE_ID/${CaseName}.log
