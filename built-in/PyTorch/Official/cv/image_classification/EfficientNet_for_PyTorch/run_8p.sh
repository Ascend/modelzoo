#!/usr/bin/env bash
source npu_set_env.sh

su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 172.17.0.1:22118 --log \"SetLogLevel(0)[error]\" --device 0"
su HwHiAiUser -c "/usr/local/Ascend/ascend-toolkit/latest/toolkit/bin/adc --host 172.17.0.1:22118 --log \"SetLogLevel(0)[error]\" --device 4"

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

taskset -c 0-128 python3.7 ${currentDir}/examples/imagenet/main.py \
    --data=/data/imagenet \
    --arch=efficientnet-b0 \
    --batch-size=2048 \
    --lr=3.2 \
    --momentum=0.9 \
    --epochs=100 \
    --autoaug \
    --amp \
    --pm=O1 \
    --loss_scale=128 \
    --val_feq=10 \
    --addr=$(hostname -I |awk '{print $1}') \
    --dist-backend=hccl \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --device_list '0,1,2,3,4,5,6,7' > ${train_log_dir}/train_8p.log 2>&1 &