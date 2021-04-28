source env.sh
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export TASK_QUEUE_ENABLE=1

#previous lr = 0.007
CUDA_VISIBLE_DEVICES=-1 python3.7.5 train_NPU.py 
                        --backbone resnet \
                        --lr 0.001 \
                        --workers 64 \
                        --epochs 100 \
                        --batch-size 8 \
                        --gpu-ids 0 \
                        --checkname a0421 \
                        --eval-interval 1 \
                        --dataset pascal 
