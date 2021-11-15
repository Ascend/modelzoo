#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改

export RANK_SIZE=1
export JOB_ID=10087
export RANK_ID_START=0
export PYTHONPATH=../transformer:$PYTHONPATH


# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改

#基础参数，需要模型审视修改
#网络名称，同目录名称
Network="Transformer_ID0633_for_TensorFlow2.X"
#训练batch_size
eval_batch_size=4
batch_size=6144
#训练step
train_steps=1000
#训练epoch
train_epochs=`expr 768 / ${batch_size}`
#学习率
learning_rate=0.000058711

#TF2.X独有，不需要修改
export NPU_LOOP_SIZE=100
export NPU_ENABLE_PERF=true
export GE_USE_STATIC_MEMORY=1

#维测参数，precision_mode需要模型审视修改
precision_mode="allow_mix_precision"
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
    
    #绑核，不需要绑核的模型删除，需要绑核的模型根据实际修改
    cpucount=`lscpu | grep "CPU(s):" | head -n 1 | awk '{print $2}'`
    cpustep=`expr $cpucount / 8`
    echo "taskset c steps:" $cpustep
    let a=RANK_ID*$cpustep
    let b=RANK_ID+1
    let c=b*$cpustep-1
    
    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path，--autotune
    nohup taskset -c $a-$c python3 ../transformer/official/nlp/transformer/transformer_main.py \
  	--data_dir=${data_path} \
 	--model_dir=${cur_path}/output/$ASCEND_DEVICE_ID/ckpt \
	--vocab_file=${data_path}/vocab.ende.32768 \
	--param_set=big \
	--train_steps=${train_steps} \
	--static_batch=true \
	--batch_size=${batch_size} \
	--steps_between_evals=100 \
	--max_length=64 \
	--bleu_source=${data_path}/newstest2014.en \
	--bleu_ref=${data_path}/newstest2014.de \
	--decode_batch_size=32 \
	--decode_max_length=97 \
	--padded_decode=False \
	--num_gpus=1 \
	--dtype=fp32 \
	--distribution_strategy='mirrored' \
	--enable_metrics_in_training=true \
	--enable_time_history=true \
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
train_accuracy=`grep eval_accuracy $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log|grep -v mlp_log|awk 'END {print $5}'|sed 's/,//g'|cut -c 1-5`
#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#############冒烟看护#########################
BatchSize=${batch_size}
#设备类型
DeviceType=`uname -m`
#用例名称
CaseName=${Network}_base_static_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#error_msg="tensorflow.python.framework.errors_impl.InternalError: Graph engine process graph failed: E89999: Inner Error!"
error_msg="op\[ConfusionTranspose\], TransposeReshapeFusionPass cannot be applied for unknown shape"

Status=`grep "${error_msg}" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | wc -l`
error_msg=`grep "${error_msg}" $cur_path/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log | tail -1`

ModelStatus="图执行FAIL"
DTS_Number="DTS2021082718985"

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ModelStatus = ${ModelStatus}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DTS_Number = ${DTS_Number}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "Status = ${Status}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "error_msg = ${error_msg}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log