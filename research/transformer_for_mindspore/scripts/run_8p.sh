#!/bin/bash
hh=$(ps -ef | grep pytest |wc -l)
if [ $hh -gt 1 ]; then
    echo "NOW THERE ARE SOME PROCESS RUNNING, ABOUT THE PYTEST, PLEASE CONTECT THE OWNER TO DEAL WITH IT!!!!!!!!"
else
echo "OK, NOW U CAN USE THIS DEVICE TO PROCESS STH"

source ./scripts/env_parallel.sh

MS_HOME=$(pwd)

rm -rf run_parallel
mkdir run_parallel
cd run_parallel

#new HCCL
export HCCL_FLAG=1
export DEPLOY_MODE=0

export RANK_SIZE=8
export MINDSPORE_HCCL_CONFIG_PATH=$MS_HOME/scripts/hccl$RANK_SIZE.json
export RANK_TABLE_FILE=$MS_HOME/scripts/hccl$RANK_SIZE.json 

for((i=0;i<$RANK_SIZE;i++))
do
    cp ../transformer ./helper$i -rf
    cd ./helper$i
    export RANK_ID=$i
    export DEVICE_ID=$i
	
    echo "start training for device $i"
    env > env$i.log
    cp ../../scripts/parallel_train.sh .
    bash parallel_train.sh > log 2>&1 &
    cd ../
done
cd ..
fi

