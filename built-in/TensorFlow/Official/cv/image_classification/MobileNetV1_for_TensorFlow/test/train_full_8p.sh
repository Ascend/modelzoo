#!/bin/bash
source env.sh
cur_path=`pwd`/../
#export ASCEND_DEVICE_ID=0

data_path="/npu/traindata/imagenet_TF"
batch_size=256

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
export RANK_SIZE=8
export RANK_TABLE_FILE=${cur_path}/8p.json
RANK_ID_START=0
for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
    echo "Device ID: $RANK_ID"
    export RANK_ID=$RANK_ID

    mkdir -p ${cur_path}/output/$ASCEND_DEVICE_ID/ckpt

    let a=RANK_ID*12
    let b=RANK_ID+1
    let c=b*12-1
    nohup taskset -c $a-$c python3.7 ${cur_path}/train.py \
        --dataset_dir=${data_path} \
        --max_epoch=300 \
        --model_name="mobilenet_v1" \
        --moving_average_decay=0.9999 \
        --label_smoothing=0.1 \
        --preprocessing_name="inception_v2" \
        --weight_decay='0.00004' \
        --batch_size=${batch_size} \
        --learning_rate_decay_type='cosine_annealing' \
        --learning_rate=0.8 \
        --optimizer='momentum' \
        --momentum='0.9' \
        --warmup_epochs=5 > $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log 2>&1 &
done
wait
end=$(date +%s)
e2etime=$(( $end - $start ))

fps=`grep -a 'logger.py' $cur_path/test/output/$ASCEND_DEVICE_ID/train_$ASCEND_DEVICE_ID.log|awk 'END {print $7}'`
step_time=`awk 'BEGIN{printf "%.2f\n",'1000'*'${batch_size}'*'${RANK_SIZE}'/'$fps'}'`

echo "Final Performance image/s : $fps"
echo "Final Performance ms/step : $step_time"
echo "Final Training Duration sec : $e2etime"