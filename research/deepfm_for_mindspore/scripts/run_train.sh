export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0
python -u ../train.py \
    --dataset_path='/opt/npu/deepFM/test_criteo_data/' \
    --do_eval=True

