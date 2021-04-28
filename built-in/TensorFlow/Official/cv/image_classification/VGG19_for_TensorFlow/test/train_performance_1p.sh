#!/bin/bash
source env.sh
cur_path=`pwd`/../
#export ASCEND_DEVICE_ID=0

data_path="/npu/traindata/imagenet_TF"
batch_size=32

if [[ $1 == --help || $1 == --h ]];then
   echo "usage:./train_full_1p.sh --data_path=data_dir --batch_size=1024 --learning_rate=0.04"
   exit 1
fi

for para in $*
do
    if [[ $para == --data_path* ]];then
      data_path=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --learning_rate* ]];then
      learning_rate=`echo ${para#*=}`
    fi
done
   
cd $cur_path

if [ -d $cur_path/test/output ];then
   rm -rf $cur_path/test/output/*
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
else
   mkdir -p $cur_path/test/output/$ASCEND_DEVICE_ID
fi
wait

start=$(date +%s)
export RANK_SIZE=1
export RANK_ID=0

python3.7 ${cur_path}/train.py \
    --batch_size=${batch_size} \
    --rank_size=1 \
    --mode=train \
    --max_train_steps=2000 \
    --iterations_per_loop=10 \
    --data_dir=${data_path} \
    --display_every=10 \
    --log_dir=$cur_path/test/output/model_1p \
    --log_name=vgg16_1p.log > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

fps=`grep -a 'FPS' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $7}'`
step_time=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'/'$fps'}'`

echo "Final Performance image/s : $fps"
echo "Final Performance ms/step : $step_time"
echo "Final Training Duration sec : $e2etime"