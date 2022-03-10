#!/bin/bash
# source env.sh
#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
# export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Nezha-large_for_TensorFlow"
#训练epoch
train_epochs=1
#训练batch_size
batch_size=64
#训练step
train_steps=1000
#学习率
learning_rate=

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

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
    --autotune               whether to enable autotune, default is False
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
    elif [[ $para == --autotune* ]];then
        autotune=`echo ${para#*=}`
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
        cp -rf $install_path/fwkacllib/data/tiling/Ascend910/custom ${autotune_dump_path}/GA/
        cp -rf $install_path/fwkacllib/data/rl/Ascend910/custom ${autotune_dump_path}/RL/
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


#训练开始时间，不需要修改
start_time=$(date +%s)
cd $cur_path/../
#进入训练脚本目录，需要模型审视修改
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup python3.7 $cur_path/../src/pretrain/run_pretraining.py \
        --bert_config_file=$cur_path/../configs/nezha_large_config.json \
        --max_seq_length=128 \
        --max_predictions_per_seq=20 \
        --train_batch_size=64 \
        --learning_rate=1e-4 \
        --num_train_steps=1000 \
        --num_warmup_steps=100 \
        --optimizer_type=lamb \
        --manual_fp16=True \
        --use_fp16_cls=True \
        --input_files_dir=${data_path}/wikipedia_128 \
        --eval_files_dir=${data_path}/cn-wiki-128 \
        --npu_bert_debug=False \
        --npu_bert_use_tdt=True \
        --do_train=True \
        --num_accumulation_steps=1 \
        --npu_bert_job_start_file= \
        --iterations_per_loop=100 \
        --save_checkpoints_steps=1000 \
        --npu_bert_clip_by_global_norm=False \
        --distributed=False \
        --npu_bert_loss_scale=0 \
        > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
    
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))


#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
#temp_FPS=`grep ' global_step/sec:' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log| awk -F "global_step/sec:" '{print $2}' | awk -F " " '{print $1}' | awk '{sum+=$1} END {print sum/NR}'`
#FPS=`echo "scale=2;${batch_size} * ${RANK_SIZE} * ${temp_FPS}"|bc`
FPS=`grep 'Throughput =      ' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | awk -F "Throughput =      " '{print $2}' | awk -F " " '{print $1}' | tail -n +2 | awk '{sum+=$1} END {print sum/NR}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
#train_accuracy=`grep -A 1 top1 $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $3}'`
#打印，不需要修改
#echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`echo "scale=2;${batch_size} * ${RANK_SIZE} * 1000 / ${FPS}"|bc`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep 'Throughput' $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print $17}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

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
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log