#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup python imagenet_main.py \
--resnet_size=101 \
--batch_size=1024 \
--num_gpus=8 \
--dtype=fp32 \
--train_epochs=90 \
--epochs_between_evals=5 \
--hooks=ExamplesPerSecondHook \
--data_dir=/home/hiscv/dataset/imagenet_TF_record \
--model_dir=./model_dir > train.log &
