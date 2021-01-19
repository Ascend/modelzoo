source npu_set_env.sh

export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`
train_log_dir=${currentDir}/result/training_1p_job_${currtime}
mkdir -p ${train_log_dir}
cd ${train_log_dir}
echo "train log path is ${train_log_dir}"
ln -s ${currentDir}/.data ${train_log_dir}/.data

python3.7 ${currentDir}/gru_1p.py \
    --workers 40 \
    --dist-url 'tcp://127.0.0.1:50000' \
    --world-size 1 \
    --npu 0 \
    --batch-size 512 \
    --epochs 10 \
    --rank 0 \
    --amp  > ./gru_1p.log 2>&1 &

