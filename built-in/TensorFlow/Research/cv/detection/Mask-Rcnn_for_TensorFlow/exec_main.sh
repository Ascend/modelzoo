#!/bin/bash
export RANK_ID=$1
export RANK_SIZE=$2
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export JOB_ID=990
export FUSION_TENSOR_SIZE=1000000000
# for producible results
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISM=1

python3 ${3}/mask_rcnn_main.py --mode=train_and_eval \
                     --train_batch_size=2 \
                     --training_file_pattern="${3}/npu_data/coco_official_2017/tfrecord/train2017*" \
                     --validation_file_pattern="${3}/npu_data/coco_official_2017/tfrecord/val2017*" \
                     --val_json_file="${3}/npu_data/coco_official_2017/annotations/instances_val2017.json" \
                     --eval_batch_size=2 \
                     --model_dir=result_npu \

sleep 2
echo "**************** train finished ***************"
#cp -r /root/ascend/log/ ./slog

