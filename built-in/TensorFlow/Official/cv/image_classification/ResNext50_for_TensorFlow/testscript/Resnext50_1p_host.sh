#!/bin/bash
#export LANG=en_US.UTF-8
currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}
# set device
device_id=0
export DEVICE_ID=${device_id}
DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8))
export DEVICE_INDEX=${DEVICE_INDEX}

#mkdir exec path
rm -rf ${currentDir}/log
mkdir -p ${currentDir}/log
mkdir -p ${currentDir}/d_solution/ckpt${DEVICE_ID}

env > ${currentDir}/log/env_${device_id}.log

#可执行npu_set_env_1p.sh文件的路径请根据实际路径修改
source /network/ResNext50_for_TensorFlow/bin/npu_set_env_1p.sh

#start exec
#可执行res50.py文件的路径请根据实际路径修改
python3.7 /network/ResNext50_for_TensorFlow/code/resnext50_train/mains/res50.py --config_file=res50_32bs_8p_host --max_train_steps=10000 --iterations_per_loop=1000   --debug=True  --eval=False  --model_dir=./  
