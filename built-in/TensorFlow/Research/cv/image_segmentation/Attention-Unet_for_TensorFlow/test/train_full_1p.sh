#!/bin/bash
cur_dir=`pwd`

#env variables
export JOB_ID=10086
export RANK_ID=0
export RANK_SIZE=1

#base param, using user config in python scripts if not config in this shell
model="aunet"
model="train"
dataset="lashan"
act=true
crop_height=112
crop_width=112
batch_size=32
num_epoch=100


if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_performance_1p.sh <args>"

    echo " "
    echo "parameter explain:
    --model                    The model you are usring. Currently supports: aunet, deep, Unet
    --mode          		   "train", "test" or "predict" mode
    --dataset        		   dataset path
    --act   		           True if sigmoind or false for softmax
    --crop_height    		   crop_height
    --crop_width               crop_width 
    --batch_size               batch_size   
    --num_epoch                train epochs.
    -h/--help                  show help message 
    "
    exit 1
fi

for para in $*
do
    if [[ $para == --model* ]];then
        model=`echo ${para#*=}`
    elif [[ $para == --mode* ]];then
        mode=`echo ${para#*=}`
    elif [[ $para == --dataset* ]];then
        dataset=`echo ${para#*=}`
    elif [[ $para == --act* ]];then
        act=`echo ${para#*=}`
    elif [[ $para == --crop_height* ]];then
        crop_height=`echo ${para#*=}`
    elif [[ $para == --crop_width* ]];then
        crop_width=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --num_epoch* ]];then
        num_epoch=`echo ${para#*=}`
    fi
done

if [[ $data_path == "" ]];then
    echo "[Error] para \"dataset\" must be confing"
    exit 1
fi


mkdir -p ${cur_dir}/output/${ASCEND_DEVICE_ID}

python3 ../mainNPU.py \
--model $model \
--mode $mode \
--dataset=$dataset \
--act $act \
--crop_height $crop_height \
--crop_width $crop_width \
--batch_size $batch_size \
--num_epoch $num_epoch  > ${cur_dir}/output/${ASCEND_DEVICE_ID}/train.log 2>&1

performance=`grep -a "Final performance FPS" ${cur_dir}/output/${ASCEND_DEVICE_ID}/train.log|awk -F" " {print $4}'`
Accuracy=`grep -a "Final accuracy" ${cur_dir}/output/${ASCEND_DEVICE_ID}/train.log|awk '{print $3}'`

echo "Final performance FPS : $performance"
echo "Final accuracy : $Accuracy"