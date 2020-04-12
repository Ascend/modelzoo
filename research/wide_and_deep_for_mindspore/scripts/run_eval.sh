export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0
python -u tools/test.py \
    --data_path './test_raw_data/' \
    --epochs 15 \
    --batch_size 10000 \
    --eval_batch_size 10000 \
    --field_size 39 \
    --vocab_size 184965 \
    --emb_dim 80 \
    --deep_layers_dim 1024 512 256 128 \
    --deep_layers_act 'relu' \
    --keep_prob 1.0 \
    --output_path './output/' \
    --ckpt_path './checkpoints/widedeep_train-15_4126.ckpt' \
    --eval_file_name 'single_eval.log' \
    --loss_file_name 'loss.log'

