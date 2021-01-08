export TASK_QUEUE_ENABLE=1
export SLOG_PRINT_TO_STDOUT=1
python3.7 ./main.py \
    --addr='xxx.xxx.xxx.xxx' \
    --seed 49  \
    --workers 16 \
    --lr 0.01 \
    --print-freq 10 \
    --eval-freq 3 \
    --dist-url 'tcp://127.0.0.1:50002' \
    --multiprocessing-distributed \
    --world-size 1 \
    --batch-size 256 \
    --device 'cuda' \
    --epochs 150 \
    --rank 0 \
    --device-list '0,1,2,3,4,5,6,7' \
    --amp \
    --opt-level 'O2' \
    --loss-scale-value 64 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --data /opt/gpu/dataset/imagenet > output_8p.log
