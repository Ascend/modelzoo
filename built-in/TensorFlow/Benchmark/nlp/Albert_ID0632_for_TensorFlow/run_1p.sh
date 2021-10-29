#!/bin/bash
python3 -m albert.run_pretraining \
    --input_file=./albert/data/tf_news_2016_zh_raw_news2016zh_1.tfrecord \
    --output_dir=out \
    --init_checkpoint= \
    --albert_config_file=./albert/albert_config/albert_config_base.json \
    --do_train \
    --do_eval \
    --train_batch_size=32 \
    --eval_batch_size=8 \
    --max_seq_length=512 \
    --max_predictions_per_seq=51 \
    --optimizer='lamb' \
    --learning_rate=0.00001375 \
    --num_train_steps=125000 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000