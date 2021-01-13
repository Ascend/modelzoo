source set_env_b023.sh



currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"

export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3

python3.7 ${currentDir}/gru_8p.py \
        --addr=$(hostname -I |awk '{print $1}') \
        --seed 123456 \
        --workers 160 \
        --print-freq 1 \
        --dist-url 'tcp://127.0.0.1:50000' \
        --dist-backend 'hccl' \
        --multiprocessing-distributed \
        --world-size 1 \
        --batch-size 4096 \
        --epoch 10 \
        --rank 0 \
        --device-list '0,1,2,3,4,5,6,7' \
        --amp \  > ./gru_8p.log 2>&1 &

