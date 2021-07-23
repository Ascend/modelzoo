#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
export RANK_SIZE=1
export JOB_ID=10087

# 数据集路径,保持为空,不需要修改
data_path=""
#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
#Network="BertBase_Google_ID0631_for_TensorFlow"
#训练batch_size
train_batch_size=32
#训练step
num_train_steps=100000
#学习率
learning_rate=1e-4

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

#其他参数
num_warmup_steps=1000
save_checkpoints_steps=5000
max_seq_length=256
max_predictions_per_seq=40
output_dir=ckpt

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                if or not over detection, default is False
    --data_dump_flag           data dump flag, default is False
    --data_dump_step           data dump step, default is 10
    --profiling                if or not profiling for performance debug, default is False
    --data_path                source data of training
    --type                     bertbase or bertlarge
    --num_warmup_steps         num_warmup_steps
    --save_checkpoints_steps   save_checkpoints_steps
    --train_batch_size         train_batch_size
    --learning_rate            learning_rate
    --num_train_steps          num_train_steps
    --bert_config_file         bert_config_file
    --output_dir               output_dir
    --max_predictions_per_seq  max_predictions_per_seq
    --max_seq_length           max_seq_length
    -h/--help                  show help message
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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --type* ]];then
        type=`echo ${para#*=}`
    elif [[ $para == --num_warmup_steps* ]];then
        num_warmup_steps=`echo ${para#*=}`
    elif [[ $para == --save_checkpoints_steps* ]];then
        save_checkpoints_steps=`echo ${para#*=}`
    elif [[ $para == --train_batch_size* ]];then
        train_batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
        learning_rate=`echo ${para#*=}`
    elif [[ $para == --num_train_steps* ]];then
        num_train_steps=`echo ${para#*=}`
    elif [[ $para == --max_predictions_per_seq* ]];then
        max_predictions_per_seq=`echo ${para#*=}`
    elif [[ $para == --max_seq_length* ]];then
        max_seq_length=`echo ${para#*=}`
    elif [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
        cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
        cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

if [[ $type != "bertbase" ]] && [[ $type != "bertlarge" ]];then
    echo "[Error] para \"type\" must be bertbase or bertlarge"
    exit 1
fi

if [[ $data_path != */ ]];then
    data_path=${data_path}/
fi

if [[ $type == bertbase ]];then
    bert_config_file=$cur_path/../configs/bert_base_12layer_config.json
else
    bert_config_file=$cur_path/../configs/bert_large_24layer_config.json
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

if [ -d ${cur_path}/output ];then
    rm -rf ${cur_path}/output/*
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID
fi

#进入训练脚本目录，需要模型审视修改
cd $cur_path/../
nohup python3 run_pretraining.py \
    --input_file=${data_path}* \
    --output_dir=${cur_path}/${output_dir} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=${bert_config_file} \
    --train_batch_size=${train_batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=${num_warmup_steps} \
    --learning_rate=${learning_rate} \
    --save_checkpoints_steps=${save_checkpoints_steps} \
    --precision_mode=${precision_mode} \
    --over_dump=${over_dump} \
    --data_dump_flag=${data_dump_flag} \
    --iterations_per_loop=1000 > ${cur_path}/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
step_sec=`grep -a 'INFO:tensorflow:global_step/sec: ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| sed -n 20p | awk '{print $2}'`
FPS=`awk 'BEGIN{printf "%d\n", '$step_sec' * '$train_batch_size'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
masked_lm_accuracy=`grep -a 'masked_lm_accuracy' $cur_path/$output_dir/eval_results.txt|awk '{print $3}'`
next_sentence_accuracy=`grep -a 'next_sentence_accuracy' $cur_path/$output_dir/eval_results.txt|awk '{print $3}'`

#打印，不需要修改
echo "Final Train masked_lm_accuracy : ${masked_lm_accuracy}"
echo "Final Train next_sentence_accuracy : ${next_sentence_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${train_batch_size}
DeviceType=`uname -m`
Network='BertBase_Google_ID0631_for_TensorFlow'_${type}
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
#TrainingTime=`expr ${train_batch_size} \* 1000 / ${FPS}`
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${train_batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "INFO:tensorflow:global_step = " ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk '{print $6}'| cut -d , -f 1 >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值，不需要修改
#ActualLoss=`grep -r 'INFO:tensorflow:Saving dict for global step' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk '{print $12}'| cut -d , -f 1`
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainMasked_lm_accuracy = ${masked_lm_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainNext_sentence_accuracy = ${next_sentence_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
