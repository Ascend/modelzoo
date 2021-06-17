source ./test/env.sh
taskset -c 0-95 python3.7 train_8p.py --img 608 608 \
                                      --data coco.yaml \
                                      --cfg cfg/yolov4_8p.cfg \
                                      --weights '' \
                                      --name yolov4 \
                                      --batch-size 256 \
                                      --epochs=300 \
                                      --amp \
                                      --opt-level O1 \
                                      --loss_scale 128 \
                                      --multiprocessing_distributed \
                                      --device 'npu' \
                                      --global_rank 0 \
                                      --device_list 0,1,2,3,4,5,6,7 \
                                      --world_size 1 \
                                      --addr $(hostname -I |awk '{print $1}') \
                                      --dist_url 'tcp://127.0.0.1:41111' \
                                      --dist_backend 'hccl' \
                                      --notest

