export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0
python -u tools/test.py \
    --data_path '/opt/npu/DCN/tf_record/' \
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
    --ckpt_path '/opt/npu/DCN_final/WideDeep-model_zoo_multinpu-40be0befd4d47f58104ba49fbcec241a28d870a4/device_1/checkpoints/widedeep_train-9_515.ckpt' \
    --eval_file_name 'single_eval.log' \
    --loss_file_name 'loss.log'

