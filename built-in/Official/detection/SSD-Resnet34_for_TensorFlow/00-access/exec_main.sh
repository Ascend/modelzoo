#!/bin/bash
export RANK_ID=$1
export RANK_SIZE=$2
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export JOB_ID=990
export FUSION_TENSOR_SIZE=1000000000

python3 ${3}/ssd_main.py --mode=train_and_eval \
                     --train_batch_size=32 \
                     --training_file_pattern="${3}/coco_official_2017/tfrecord/train2017*" \
                     --resnet_checkpoint="${3}/resnet34_pretrain/model.ckpt-28152" \
                     --validation_file_pattern="${3}/coco_official_2017/tfrecord/val2017*" \
                     --val_json_file="${3}/coco_official_2017/annotations/instances_val2017.json" \
                     --eval_batch_size=32 \
                     --model_dir=result_npu


sleep 2
echo "**************** train finished ***************"
cp /var/log/npu/slog/host-0/* ./slog
cp /var/log/npu/slog/device-$DEVICE_ID/* ./slog
cp /var/log/npu/slog/device-os-$DEVICE_ID/* ./slog

