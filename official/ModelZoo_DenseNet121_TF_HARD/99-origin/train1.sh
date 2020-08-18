#!/bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#export RANK_ID=${device_id}
echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

#echo "${currentDir}"

env > ${currentDir}/env_${device_id}.log

#mkdir exec path
mkdir -p ${currentDir}/${device_id}
rm -rf ${currentDir}/${device_id}/*
cd ${currentDir}/${device_id}

#start exec
#python3.7 /opt/npu/z00438116/tf/NetDownload/Resnet50_HC/code/vgg16_train/mains/vgg16.py --config_file=vgg16_64bs_2p_host --max_train_steps=1000 --iterations_per_loop=100 --debug=True --eval=False --model_dir=${PWD}/ckpt${DEVICE_ID} > ${currentDir}/train_${device_id}.log 2>&1
#python3.7 /raid5/z00438116/vgg16_tf/train.py --config_file vgg16_config_2p_host > ./train_${device_id}.log 2>&1
python3.7 ${currentDir}/train.py --config_file densenet_config_1p_npu > ./train_${device_id}.log 2>&1
#python3.7 ./train.py --config_file vgg16_config_8p_host > ./train_${device_id}.log 2>&1
#python3.7 /opt/npu/z00438116/vgg16/vgg16_tf_v2/train.py --config_file vgg16_config_8p_host > ./train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/train_${device_id}.log
fi

