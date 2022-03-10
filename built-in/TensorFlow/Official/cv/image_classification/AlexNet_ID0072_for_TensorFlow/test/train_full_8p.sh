#!/bin/bash

#当前路径,不需要修改
cur_path=`pwd`

#集合通信参数,不需要修改
#保证rank table file 文件rank_table_8p.json存放在和test同级的configs目录下
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/../configs/8p.json
export JOB_ID=10087
RANK_ID_START=0

# 数据集路径,保持为空,不需要修改
data_path=""

#设置默认日志级别,不需要修改
#export ASCEND_GLOBAL_LOG_LEVEL=3

#基础参数 需要模型审视修改
#网络名称，同目录名称
Network="AlexNet_ID0072_for_TensorFlow"

#训练batch_size
batch_size=128

#维持参数，以下不需要修改
over_dump=False
data_dump_flag=False
data_dump_step="10"
profiling=False
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
    elif [[ $para == --bind_core* ]]; then
        bind_core=`echo ${para#*=}`
        name_bind="_bindcore"
    fi
done

#校验是否传入data_path,不需要修改
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
	
	
    exit 1
fi


#autotune时，先开启autotune执行单P训练，不需要修改
if [[ $autotune == True ]]; then
    sh -x train_full_1p.sh --autotune=$autotune --data_path=$data_path
    wait
    autotune=False
	 export autotune=False
	 
	export RANK_SIZE=8
    export RANK_TABLE_FILE=${cur_path}/../configs/8p.json
    export JOB_ID=10087
    RANK_ID_START=0
	unset TE_PARALLEL_COMPILER
	 
fi

#训练开始时间，不需要修改
start_time=$(date +%s)

#进入训练脚本目录，需要模型审视修改

for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    #设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    export DEVICE_ID=$RANK_ID
	DEVICE_INDEX=$RANK_ID
    export DEVICE_INDEX=${DEVICE_INDEX}

    #创建DeviceID输出目录，不需要修改
    if [ -d ${cur_path}/output/${ASCEND_DEVICE_ID} ];then
        rm -rf ${cur_path}/output/${ASCEND_DEVICE_ID}
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    else
        mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt
    fi

	# 绑核，不需要的绑核的模型删除，需要模型审视修改
    corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
    let a=RANK_ID*${corenum}/${RANK_SIZE}
    let b=RANK_ID+1
    let c=b*${corenum}/${RANK_SIZE}-1

    #执行训练脚本，以下传参不需要修改，其他需要模型审视修改
    #--data_dir, --model_dir, --precision_mode, --over_dump, --over_dump_path，--data_dump_flag，--data_dump_step，--data_dump_path，--profiling，--profiling_dump_path
    if [ "x${bind_core}" != x ];then
        bind_core="taskset -c $a-$c"
    fi
python3.7 ${cur_path}/../train.py --rank_size=8 \
                      --epochs_between_evals=1 \
                      --mode=train \
        	            --max_epochs=150 \
                      --iterations_per_loop=100 \
        	            --batch_size=${batch_size} \
        	            --data_dir=${data_path} \
        	            --lr=0.06 \
                      --checkpoint_dir=./model_8p \
        	            --log_dir=./model_8p > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
						


done
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
#设置环境变量，不需要修改
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID
    export ASCEND_DEVICE_ID=$RANK_ID
    ASCEND_DEVICE_ID=$RANK_ID

    export DEVICE_ID=$RANK_ID
	DEVICE_INDEX=$RANK_ID
    export DEVICE_INDEX=${DEVICE_INDEX}
python3 ${cur_path}/../train.py --rank_size=8 \
                      --epochs_between_evals=1 \
                      --mode=evaluate \
        	            --max_epochs=150 \
                      --iterations_per_loop=100 \
        	            --batch_size=${batch_size} \
        	            --data_dir=${data_path} \
        	            --lr=0.06 \
                      --checkpoint_dir=./model_8p \
        	            --log_dir=./model_8p >> ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
done
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能ms/step,需要模型审视修改
step_sec=`grep FPS  ${cur_path}/output/0/train_0.log|awk 'END {print $3}'|awk -F ":" 'END {print $2}'|awk -F "," 'END {print $1}'`
#打印，不需要修改
echo "Final Performance ms/step : $step_sec"


#打印，不需要修改
echo "Final Training Duration sec : $e2e_sec"
#输出训练精度,需要模型审视修改
train_accuracy=`grep -A 3 "top1" ${cur_path}/output/0/train_0.log|tail -1|awk 'END {print $3}'`
#打印，不需要修改
echo "Final train_accuracy is ${train_accuracy}"
echo "E2E training Duration sec: $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}${name_bind}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据
#吞吐量，不需要修改
ActualFPS=${step_sec}
#单迭代训练时长,需要模型审视修改
TrainingTime=`expr ${batch_size} \* 1000 / ${step_sec}`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
`grep total_loss ${cur_path}/output/0/train_0.log|awk 'END {print $5}'|tr -d , >> $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`

#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' $cur_path/output/$ASCEND_DEVICE_ID/train_${CaseName}_loss.txt`
TrainAccuracy=$train_accuracy
#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainAccuracy = ${TrainAccuracy}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/output/$ASCEND_DEVICE_ID/${CaseName}.log