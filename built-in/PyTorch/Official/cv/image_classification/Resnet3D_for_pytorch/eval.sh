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


python3 main.py \
    --video_path /data/hj/resnet3d/hmdb51_jpg \
    --annotation_path /data/hj/resnet3d/hmdb51_json/hmdb51_1.json \
    --result_path outputs \
    --dataset hmdb51 \
    --resume_path outputs/save_200.pth \
    --model_depth 18 \
    --n_classes 51 \
    --n_threads 4 \
    --no_train \
    --no_val \
    --inference \
    --output_topk 5 \
    --inference_batch_size 1 \
    --device_list '4'

python3 -m util_scripts.eval_accuracy /data/hj/resnet3d/hmdb51_json/hmdb51_1.json outputs/val.json -k 1 --ignore
