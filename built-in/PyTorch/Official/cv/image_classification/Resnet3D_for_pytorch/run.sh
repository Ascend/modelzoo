source ./env_b031.sh
source ./env_new.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export COMBINED_ENABLE=1
export DYNAMIC_OP="ADD#MUL"
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

nohup python3 main.py \
    --video_path /data/hj/resnet3d/hmdb51_jpg \
    --annotation_path /data/hj/resnet3d/hmdb51_json/hmdb51_1.json \
    --result_path outputs \
    --dataset hmdb51 \
    --n_classes 51 \
    --n_pretrain_classes 700 \
    --pretrain_path r3d18_K_200ep.pth \
    --ft_begin_module fc \
    --model resnet \
    --model_depth 18 \
    --batch_size 128 \
    --n_threads 16 \
    --checkpoint 5 \
    --amp_cfg \
    --opt_level O2 \
    --loss_scale_value 1024 \
    --device_list '4' \
    --manual_seed 1234 \
    --learning_rate 0.01 \
    --tensorboard &
