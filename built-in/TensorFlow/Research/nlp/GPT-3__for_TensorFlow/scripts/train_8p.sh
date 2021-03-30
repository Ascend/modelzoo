
#!/bin/sh
CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}

cd ..
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
num_cpus=$(getconf _NPROCESSORS_ONLN)
num_cpus_per_device=$((num_cpus/8))

start_id=$((num_cpus_per_device*device_id))
end_id=$((num_cpus_per_device*device_id+num_cpus_per_device-1))


#taskset -c ${start_id}-${end_id} python3.7 ${EXEC_DIR}/main.py > ./train_${device_id}.log 2>&1
python3.7 ${EXEC_DIR}/main.py > ./train_${device_id}.log 2>&1

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${device_id}.log
fi
