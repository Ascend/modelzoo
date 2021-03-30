#!/bin/bash

#EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
#cd $EXEC_DIR

#cd ..
EXEC_DIR=$(pwd)
echo $EXEC_DIR
cd  ${EXEC_DIR}
RESULTS=results/8p


#mkdir exec path
#mkdir -p ${EXEC_DIR}/${RESULTS}/${device_id}
#rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*
#cd ${EXEC_DIR}/${RESULTS}/${device_id}

device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ./train_${device_id}.log
    exit
else
    export DEVICE_ID=${device_id}
fi

export RANK_ID=${device_id}
echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID


env > ${EXEC_DIR}/${RESULTS}/env_${device_id}.log

#start exec
python3.7 ${EXEC_DIR}/nmt.py  \
  --output_dir=${EXEC_DIR}/${RESULTS}/${device_id} \
  --batch_size=1024 \
  --learning_rate=2e-3 \
  --use_amp=False  > ./train_${device_id}.log 2>&1 &


if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
fi

