#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=1
unset RANK_TABLE_FILE
#export RANK_TABLE_FILE=${cur_path}/../configs/rank_table_8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
export ASCEND_GLOBAL_LOG_LEVEL=3

#网络名称，同目录名称
Network="AlexNet_for_TensorFlow"
#训练batch_size
batch_size=256
#学习率
learning_rate=0.015
#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
#参数校验，不需要修改
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
#help info

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
		autotune=True
#开autotune特有环境变量
		export autotune=True
		export REPEAT_TUNE=True
		export ASCEND_DEVICE_ID=0
		export ENABLE_TUNE_BANK=True
		export TE_PARALLEL_COMPILER=32
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



         # sed -i 's/n_epoches = 1/n_epoches = 20/g' ../configs/config.py 

        #  sed -i 's/iteration_per_loop = 1/iteration_per_loop = 10/g' ../configs/config.py
		  

    exit 1
	
fi

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
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

EXEC_DIR=$(pwd)
RESULTS=results/1p

mkdir -p ${EXEC_DIR}/${RESULTS}/${ASCEND_DEVICE_ID}	
	
rm -rf ${EXEC_DIR}/${RESULTS}/${ASCEND_DEVICE_ID}/*

cd ${EXEC_DIR}/${RESULTS}/${ASCEND_DEVICE_ID}

env > ${EXEC_DIR}/${RESULTS}/env_${ASCEND_DEVICE_ID}.log


python3.7 ${EXEC_DIR}/../train.py --rank_size=1 \
	--iterations_per_loop=100 \
	--batch_size=${batch_size} \
	--data_dir=${data_path} \
	--mode=train \
	--checkpoint_dir=${EXEC_DIR}/${RESULTS}/${ASCEND_DEVICE_ID}/model_1p/ \
	--lr=0.015 \
	--log_dir=./model_1p > ./train_${ASCEND_DEVICE_ID}.log 2>&1 

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${ASCEND_DEVICE_ID}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${ASCEND_DEVICE_ID}.log
fi


done





