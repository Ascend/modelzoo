#!/bin/bash

# temporally comment this for restoring from ckpt
#rm -rf ./model_dir_8p_xla_amp/*

# 1p
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TF_ENABLE_AUTO_MIXED_PRECISION=1
python test.py --config_file densenet_config_8p_gpu
#horovodrun -np 8 -H localhost:8 python test.py --config_file vgg16_config_8p_gpu
