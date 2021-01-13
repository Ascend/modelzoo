source set_env_b023.sh


export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3

nohup python3.7 gru_8p.py
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

