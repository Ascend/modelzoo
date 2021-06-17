#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
export RANK_ID_START=0


# 数据集路径,保持为空,不需要修改
data_path=""

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="BertLarge_TF2.X_for_TensorFlow"

#训练batch_size
eval_batch_size=4
batch_size=16
#训练step
train_steps=8000
#训练epoch
train_epochs=`expr 768 / ${batch_size}`
#学习率
learning_rate=0.000058711

#TF2.X独有，不需要修改
#export NPU_LOOP_SIZE=${train_steps}
export NPU_LOOP_SIZE=1

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_fp32_to_fp16"
#维持参数，以下不需要修改
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
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

init_ckpt_path=${data_path}/'tf2_ckpt/model.ckpt-28252'  #need modify to actual path
train_files_path=${data_path}/'train/*'  #need modify to actual path
eval_files_path=${data_path}/'eval/eval.tfrecord'  #need modify to actual path

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $ASCEND_DEVICE_ID"
    export RANK_ID=$RANK_ID
    
    
    
    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate}
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate}
    fi

    # 绑核，不需要的绑核的模型删除，需要的模型审视修改
    let a=RANK_ID*12
    let b=RANK_ID+1
    let c=b*12-1
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup taskset -c $a-$c python3 ../bert/run_pretraining.py \
  	--all_reduce_alg=nccl \
 	 --bert_config_file=../configs/bert_base_config.json \
  	--beta_1=0.91063 \
  	--beta_2=0.96497 \
  	--device_warmup=False \
  	--do_eval=True \
  	--dtype=fp16 \
  	--eval_batch_size=${eval_batch_size} \
 	--train_files=${train_files_path} \
  	--eval_files=${eval_files_path} \
  	--learning_rate=${learning_rate} \
  	--loss_scale=dynamic \
  	--max_predictions_per_seq=76 \
  	--max_seq_length=512 \
  	--model_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt_${learning_rate} \
  	--num_accumulation_steps=1 \
	--distribution_strategy=one_device \
  	--num_gpus=1 \
 	--num_steps_per_epoch=8000 \
  	--num_train_epochs=${train_epochs} \
  	--optimizer_type=lamb \
  	--scale_loss=False \
	--stop_threshold=0.95 \
  	--steps_between_eval=2000 \
  	--steps_per_loop=${NPU_LOOP_SIZE} \
  	--stop_steps=60000 \
	--enable_checkpoint_and_summary=True \
  	--train_batch_size=${batch_size} \
  	--verbosity=0 \
  	--warmup_steps=0 \
	--precision_mode=${precision_mode} \
        --over_dump=${over_dump} \
        --over_dump_path=${over_dump_path} \
        --data_dump_flag=${data_dump_flag} \
        --data_dump_step=${data_dump_step} \
        --data_dump_path=${data_dump_path} \
	--profiling=${profiling} \
	--profiling_dump_path=${profiling_dump_path} > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done 
wait
#--init_checkpoint=${init_ckpt_path} \

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
single_batch_step_sec=`grep TimeHistory  $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $8}'`
FPS=`awk 'BEGIN{printf "%.2f\n",'${single_batch_step_sec}'*'${batch_size}'}'`
#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "Train Step" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|awk 'END {print $34}'|cut -c 1-5`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#############冒烟看护#########################
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${FPS}
#单迭代训练时长，不需要修改
TrainingTime=`awk 'BEGIN{printf "%.2f\n",'${BatchSize}'*'${RANK_SIZE}'*1000/'${FPS}'}'`

##获取Loss
#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中
grep "Train Step" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk '{print$11}' >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt

#最后一个迭代loss值
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RANK_SIZE = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log

