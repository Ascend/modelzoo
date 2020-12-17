#!/bin/bash
#export LANG=en_US.UTF-8
set -x
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
RANK_ID_START=0
RANK_SIZE=8
SAVE_PATH=training
BASE_PATH=`pwd`
echo $BASE_PATH

for ((RANK_ID = $RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
do
echo 
TMP_PATH=$SAVE_PATH/D$RANK_ID
mkdir -p $TMP_PATH
cd $TMP_PATH
source /network/ResNext50_for_TensorFlow/bin/npu_set_env.sh $RANK_ID $RANK_SIZE

#start exec
#可执行res50.py文件的路径请根据实际路径修改
python3.7 /network/ResNext50_for_TensorFlow/code/resnext50_train/mains/res50.py --config_file=res50_32bs_8p_host --max_train_steps=10000 --iterations_per_loop=1000   --debug=True  --eval=False  --model_dir=./  2>&1 | tee train_$RANK_ID.log &

cd -
done
