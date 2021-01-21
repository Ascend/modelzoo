
#!/bin/sh
CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}

EXEC_DIR=$(pwd)
RESULTS=results/8p


device_id=$1
if  [ x"${device_id}" = x ] ;
then
    echo "turing train fail" >> ${EXEC_DIR}/results/train_${device_id}.log
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

#mkdir exec path
mkdir -p ${EXEC_DIR}/${RESULTS}/${device_id}
rm -rf ${EXEC_DIR}/${RESULTS}/${device_id}/*
cd ${EXEC_DIR}/${RESULTS}/${device_id}


#mkdir -p ${EXEC_DIR}/results

env > ${EXEC_DIR}/results/env_${device_id}.log


#start exec
local_path='./model1'
train_folder='/home/t00495118/processed_hisi'
epoch=1
steps_per_epoch=1
lr=8e-4
log_every_n_steps=1
multi_npu=1
rank_size=8
rank_id=$device_id

nohup python3 -u $EXEC_DIR/train_estimator_npu.py --local_path=$local_path \
	--train_folder=$train_folder \
	--epoch=$epoch \
	--steps_per_epoch=$steps_per_epoch \
	--lr=$lr \
	--log_every_n_steps=$log_every_n_steps \
	--multi_npu=$multi_npu \
	--rank_id=$rank_id \
	--rank_size=$rank_size > $EXEC_DIR/'test'$device_id'.out' &


if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
fi
