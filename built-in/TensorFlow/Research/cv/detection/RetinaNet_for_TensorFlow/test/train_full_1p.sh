#!/bin/bash
cur_dir=`pwd`

#env variables
export JOB_ID=10086
export RANK_ID=0
export RANK_SIZE=1

#base param, using user config in python scripts if not config in this shell
batch_size=24
learning_rate=0.1
steps=100000
#epochs=0
ckpt_count=99999

#npu param
precision_mode="allow_fp32_to_fp16"
loss_scale_value=1.0
loss_scale_type="static"
over_dump=False
data_dump_flag=0
data_dump_step=10
profiling=False
random_remove=False
data_path="../data_path/"

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"

    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --loss_scale_value		   loss scale value, default is 1.0
    --loss_scale_type		   loss scale type (dynamic/static), default is static
    --over_dump		           if or not over detection, default is False
    --data_dump_flag		   data dump flag, default is 0
    --data_dump_step		   data dump step, default is 10
    --profiling		           if or not profiling for performance debug, default is False
    --random_remove		       remove train random treament, default is False
    --batch_size		       train batch size 
    --learning_rate		       learning rate
    --steps		               training steps
    --data_path		           source data of training
    --ckpt_count		       save checkpoint counts
    -h/--help		           show help message
    "
    exit 1
fi

for para in $*
do
    if [[ $para == --precision_mode* ]];then
        precision_mode=`echo ${para#*=}`
    elif [[ $para == --loss_scale_value* ]];then
        loss_scale_value=`echo ${para#*=}`
    elif [[ $para == --loss_scale_type* ]];then
        loss_scale_type=`echo ${para#*=}`
    elif [[ $para == --over_dump* ]];then
        over_dump=`echo ${para#*=}`
    elif [[ $para == --data_dump_flag* ]];then
        data_dump_flag=`echo ${para#*=}`
    elif [[ $para == --data_dump_step* ]];then
        data_dump_step=`echo ${para#*=}`
    elif [[ $para == --profiling* ]];then
        profiling=`echo ${para#*=}`
    elif [[ $para == --random_remove* ]];then
        random_remove=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
        learning_rate=`echo ${para#*=}`
    elif [[ $para == --steps* ]];then
        steps=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --ckpt_count* ]];then
        ckpt_count=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

mkdir -p ${cur_dir}/output/${ASCEND_DEVICE_ID}

python3 ../train.py \
--steps=$steps \
--ckpt_count=$ckpt_count \
--data_path=$data_path > ${cur_dir}/output/${ASCEND_DEVICE_ID}/train.log 2>&1

performance=`grep "Final Performance TotalTimeToTrain" ${cur_dir}/output/${ASCEND_DEVICE_ID}/train.log|awk '{print $5}'`
Accuracy=`grep "Final Accuracy total_loss" ${cur_dir}/output/${ASCEND_DEVICE_ID}/train.log|awk '{print $5}'`
echo "Final Performance TotalTimeToTrain(s) : $performance"
echo "Final Accuracy total_loss : $Accuracy"