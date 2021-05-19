source pt_set_env.sh
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export TASK_QUEUE_ENABLE=1

IP=$(hostname -I |awk '{print $1}')
#previous lr = 0.007
python3.7 -m train_NPU_8p \
                        --backbone resnet \
                        --lr 0.001 \
                        --workers 64 \
                        --epochs 100 \
                        --batch-size 32 \
                        --gpu-ids 0 \
                        --checkname deeplab-resnet \
                        --eval-interval 1 \
                        --dataset pascal \
                        --multiprocessing_distributed \
                        --rank 0 \
                        --addr $IP \
                        --world_size 8 \
                        --device_num 8 \
                        --device_list 0,1,2,3,4,5,6,7 


