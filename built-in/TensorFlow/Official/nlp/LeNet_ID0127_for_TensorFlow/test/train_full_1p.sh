#!/bin/bash
cur_path=`pwd`/../

if [ "x${ASCEND_DEVICE_ID}" == "x" ];then
    ASCEND_DEVICE_ID=0
fi

#参数配置
#base param, using user config in python scripts if not config in this shell
batch_size=64
learning_rate=0
steps=1000
epochs=5
ckpt_count=

#npu param
precision_mode="allow_mix_precision"
loss_scale_value=0
loss_scale_flag=0
over_dump=False
data_dump_flag=0
data_dump_step="0|5|10"
profiling=False
random_remove=False
data_path="../MNIST"

if [[ $1 == --help || $1 == -h ]];then
    echo"usage:./train_full_1p.sh <args>"

    echo " "
    echo "parameter explain:
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --loss_scale_value		   loss scale value, default is 0
    --loss_scale_flag		   loss scale flag, default is 0
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
    elif [[ $para == --loss_scale_flag* ]];then
        loss_scale_flag=`echo ${para#*=}`
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

if [ -d $cur_path/test/output ];then
	rm -rf $cur_path/test/output/*
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
	mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
  
#############执行训练#########################
python3 ../LeNet.py \
	--precision_mode=$precision_mode \
	--loss_scale_value=$loss_scale_value \
	--loss_scale_flag=$loss_scale_flag \
	--over_dump=$over_dump \
	--data_dump_flag=$data_dump_flag \
	--data_dump_step=$data_dump_step \
	--profiling=$profiling \
	--random_remove=$random_remove \
	--data_path=$data_path \
	--steps=$steps \
    > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log

#############结果处理#########################
echo "-------Test result ----------"
#step_ms=`grep -a 'Final Performance ms/step : ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $4}'`
#e2e_sec=`grep -a 'Final Training Duration sec : ' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $5}'`
#echo "Final Performance ms/step : $step_ms"
#echo "Final Training Duration sec : $e2e_sec"
cat $cur_path/test/output/performance_precision.txt