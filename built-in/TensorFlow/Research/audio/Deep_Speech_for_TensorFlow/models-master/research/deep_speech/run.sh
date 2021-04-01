#!/bin/bash
python3 deep_speech.py \
        --train_data_dir=/home/w00348617/Deep_Speech/data/librivox-test.csv \
        --eval_data_dir=/home/w00348617/Deep_Speech/data/librivox-test.csv \
        --num_gpus=1 \
        --wer_threshold=0.23 \
        --seed=1 |& tee train.log
