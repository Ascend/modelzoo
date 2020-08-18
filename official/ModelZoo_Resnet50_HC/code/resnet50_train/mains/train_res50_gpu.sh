#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

python3.5 res50.py --config_file res50_baseline_gpu
