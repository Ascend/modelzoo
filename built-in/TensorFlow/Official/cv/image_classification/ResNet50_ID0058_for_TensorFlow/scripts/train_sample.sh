#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

# set device
device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail"
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/d_solution/ckpt${DEVICE_ID}

env > ${currentDir}/log/env_${device_id}.log

#start exec
python3.7 ../src/mains/res50.py \
    --config_file=res50_256bs_8p \
    --max_train_steps=1000 \
    --iterations_per_loop=100 \
    --debug=True \
    --eval=True \
    --data_path=/data/imagenet/ \
    --model_dir=${currentDir}/d_solution/ckpt${DEVICE_ID} > ${currentDir}/log/train_${device_id}.log 2>&1
