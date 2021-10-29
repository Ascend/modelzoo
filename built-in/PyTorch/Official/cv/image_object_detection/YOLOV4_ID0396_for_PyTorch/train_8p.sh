source ./test/env.sh

for i in $(seq 0 7)
do
    if [ $(uname -m) = "aarch64" ]
    then
    let p_start=0+24*i
    let p_end=23+24*i
    taskset -c $p_start-$p_end python3.7 train_8p.py --img 608 608 \
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
                                          --global_rank $i \
                                          --device_list 0,1,2,3,4,5,6,7 \
                                          --world_size 1 \
                                          --addr $(hostname -I |awk '{print $1}') \
                                          --dist_url 'tcp://127.0.0.1:41111' \
                                          --dist_backend 'hccl' \
                                          --notest &
    else
        python3.7 train_8p.py --img 608 608 \
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
                   --global_rank $i \
                   --device_list 0,1,2,3,4,5,6,7 \
                   --world_size 1 \
                   --addr $(hostname -I |awk '{print $1}') \
                   --dist_url 'tcp://127.0.0.1:41111' \
                   --dist_backend 'hccl' \
                   --notest &
    fi
done

