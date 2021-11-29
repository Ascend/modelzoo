#!/bin/bash
#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=1
batch_size=32
#export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""
RANK_SIZE=1
epoch=1

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="Bertsquad_ID0495_for_TensorFlow"

#export RANK_ID=npu8p
#export SLOG_PRINT_TO_STDOUT=0

#维持参数，不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
autotune=False

# 帮助信息，不需要修改
if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_8p.sh <args>"
    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --autotune                 whether to enable autotune, default is False
    --data_path		           source data of training
    -h/--help		           show help message
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
    #开autotune特有环境变量
		autotune=True
		export autotune=True
		export REPEAT_TUNE=True
		#export ASCEND_DEVICE_ID=0
		export ENABLE_TUNE_BANK=True
		export TE_PARALLEL_COMPILER=32
        mv $install_path/fwkacllib/data/rl/Ascend910/custom $install_path/fwkacllib/data/rl/Ascend910/custom_bak
        mv $install_path/fwkacllib/data/tiling/Ascend910/custom $install_path/fwkacllib/data/tiling/Ascend910/custom_bak
        autotune_dump_path=${cur_path}/output/autotune_dump
        mkdir -p ${autotune_dump_path}/GA
        mkdir -p ${autotune_dump_path}/rl
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi


vocab_file=${data_path}/model/vocab.txt
bert_config_file=${data_path}/model/bert_config.json
init_checkpoint=${data_path}/model/bert_model.ckpt
train_file=${data_path}/dataset/train-v1.1_small.json
predict_file=${data_path}/dataset/dev-v1.1.json

rm -rf $cur_path/../output

cp $cur_path/run_squad.py $cur_path/../
#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改
cd $cur_path/..
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export DEVICE_ID=$ASCEND_DEVICE_ID
	DEVICE_INDEX=$RANK_ID
    RANK_INDEX=$RANK_ID
    export RANK_INDEX=$RANK_ID
    export DEVICE_INDEX=${DEVICE_INDEX}
    
    #创建DeviceID输出目录，不需要修改
    if [ -d $cur_path/output/$ASCEND_DEVICE_ID ];then
        rm -rf $cur_path/output/$ASCEND_DEVICE_ID
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

    #执行训练脚本，需要模型审视修改
	nohup python3.7 run_squad.py \
        --rank_size=$RANK_SIZE \
        --batch_size=$batch_size \
	    --data_path=$data_path \
        --vocab_file=$vocab_file \
        --bert_config_file=$bert_config_file \
        --init_checkpoint=$init_checkpoint \
        --train_file=$train_file \
        --do_predict=True \
        --do_train=True \
        --predict_file=$predict_file \
        --num_train_epochs=1 \
        --learning_rate=3e-5 \
        --max_seq_length=384 \
        --doc_stride=128 \
        --output_dir=./output \
		> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
        #--precision_mode=${precision_mode} \
        #--data_dump_flag=${data_dump_flag} \
        #--data_dump_step=${data_dump_step} \
        #--data_dump_path=${data_dump_path} \
        #--profiling=${profiling} \
        #--profiling_dump_path=${profiling_dump_path} \
        #--autotune=${autotune} \
        #rm -rf $cur_path/../output/*
done 
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#E2E训练端到端时长，直接计算，不需要修改
echo "E2E training Duration sec: $e2e_time"

#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

#获取性能数据
fps=`grep "global_step/sec:" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F 'global_step/sec:' '{print $2}'|awk 'END {print $1}'`

ActualFPS=`echo "scale=2;${fps} * ${batch_size}"|bc`
temp1=`echo "1000 * ${batch_size}"|bc`
TrainingTime=`echo "scale=2;${temp1} / ${ActualFPS}"|bc`

ActualLoss=`grep "loss =" $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk -F 'loss =' '{print $2}'|awk 'END {print $1}'|tr -d ,`

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
sed -i -e '/ModuleNotFoundError/d' $cur_path/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log