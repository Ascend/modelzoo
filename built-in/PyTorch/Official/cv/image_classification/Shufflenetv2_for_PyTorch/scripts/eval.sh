#!/usr/bin/env bash
source npu_set_env.sh
export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1

device_id_list=0,1,2,3,4,5,6,7

currentDir=$(cd "$(dirname "$0")";pwd)/..
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_8p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

python3.7 ${currentDir}/8p_main_med.py \
    --data=/data/imagenet \
    --addr=$(hostname -I |awk '{print $1}') \
    --seed=49  \
    --workers=184 \
    --learning-rate=4 \
    --print-freq=1 \
    --eval-freq=5 \
    --arch=shufflenet_v2_x1_0  \
    --dist-url='tcp://127.0.0.1:50000' \
    --dist-backend='hccl' \
    --multiprocessing-distributed \
    --world-size=1 \
    --batch-size=8192 \
    --epochs=240 \
    --warm_up_epochs=5 \
    --device_num=8 \
    --rank=0 \
    --amp \
    --momentum=0 \
    --device-list=${device_id_list} \
    --evaluate \
    --benchmark 0 > ./shufflenetv2_8p.log 2>&1 &