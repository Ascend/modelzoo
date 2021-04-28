source env.sh
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
export TASK_QUEUE_ENABLE=1

#previous lr = 0.007
CUDA_VISIBLE_DEVICES=-1 python3.7.5 -m train_NPU_8p 
                        --backbone resnet \
                        --lr 0.001 \
                        --workers 4 \
                        --epochs 100 \
                        --batch-size 8 \
                        --gpu-ids 0 \
                        --checkname deeplab-resnet \
                        --eval-interval 1 \
                        --dataset pascal \
                        --multiprocessing_distributed \
                        --rank 0 \
                        --addr 127.0.0.1 \
                        --world_size 8 \
                        --device_num 8 \
                        --device_list 0,1,2,3,4,5,6,7 


