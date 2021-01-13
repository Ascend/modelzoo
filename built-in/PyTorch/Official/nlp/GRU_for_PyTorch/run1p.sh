source set_env_b023.sh


export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3

nohup python3.7 gru_1p.py
    --workers 40 \
    --dist-url 'tcp://127.0.0.1:50000' \
    --world-size 1 \
    --npu 0 \
    --batch-size 512 \
    --epochs 10 \
    --rank 0 \
    --amp \ > ./gru_1p.log 2>&1 &

