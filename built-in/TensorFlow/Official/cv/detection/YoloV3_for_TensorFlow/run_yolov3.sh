#!/bin/bash
rm -rf Onnxgraph
rm -rf Partition
rm -rf OptimizeSubGraph
rm -rf Aicpu_Optimized
rm *txt
rm -rf result_$RANK_ID

# params set
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --save_dir* ]];then
        save_dir=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --MODE* ]];then
        MODE=`echo ${para#*=}`
    elif [[ $para == --RANK_ID* ]];then
        RANK_ID=`echo ${para#*=}`
        export RANK_ID=$RANK_ID
    elif [[ $para == --RANK_SIZE* ]];then
        RANK_SIZE=`echo ${para#*=}`
        export RANK_SIZE=$RANK_SIZE
    elif [[ $para == --MAIN_PATH* ]];then
        MAIN_PATH=`echo ${para#*=}`
    fi
done

export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
#export RANK_TABLE_FILE=rank_table.json
export JOB_ID=123678
export FUSION_TENSOR_SIZE=1000000000

corenum=`cat /proc/cpuinfo |grep "processor"|wc -l`
let a=RANK_ID*${corenum}/${RANK_SIZE}
let b=RANK_ID+1
let c=b*${corenum}/${RANK_SIZE}-1

#sleep 5
taskset -c  $a-$c python3 $MAIN_PATH/train.py \
--mode $MODE --train_file $data_path --save_dir $save_dir --batch_size $batch_size

mkdir graph
mv *.txt graph
mv *.pbtxt graph
