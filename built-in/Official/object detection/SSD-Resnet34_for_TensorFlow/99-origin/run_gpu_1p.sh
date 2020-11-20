#!/bin/bash

python ssd_main.py --mode=train_and_eval \
                     --train_batch_size=32 \
                     --training_file_pattern="/home/hiscv/dataset/coco_official_2017/tfrecord/train2017*" \
                     --resnet_checkpoint=/home/hiscv/lijing/ssd_model_0808/resnet34/model.ckpt-28152 \
                     --validation_file_pattern="/home/hiscv/dataset/coco_official_2017/tfrecord/val2017*" \
                     --val_json_file="/home/hiscv/dataset/coco_official_2017/annotations/instances_val2017.json" \
                     --eval_batch_size=32 \
                     --gpu_num=1 \
                     --model_dir=result

