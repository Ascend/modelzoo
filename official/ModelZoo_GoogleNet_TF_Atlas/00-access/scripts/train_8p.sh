#! /bin/sh
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

PWD=${currentDir}
dname=$(dirname "$PWD")

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
mkdir -p ${currentDir}/result/8p/${device_id}
#rm -rf ${currentDir}/result/8p/${device_id}/*
cd ${currentDir}/result/8p/${device_id}

#start exec
python3.7 ${dname}/train.py --rank_size=8 \
    --mode=train_and_evaluate \
    --max_epochs=200 \
    --iterations_per_loop=10 \
    --batch_size=64 \
    --display_every=100 \
    --data_dir=/opt/npu/imagenet_data \
    --lr=0.01 \
    --log_dir=./model \
    --eval_dir=./model \
    --log_name=googlenet.log > ${currentDir}/result/8p/train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${currentDir}/result/8p/train_${device_id}.log
else
    echo "turing train fail" >> ${currentDir}/result/8p/train_${device_id}.log
fi

