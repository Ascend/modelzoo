export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0
python -u ../test.py \
    --dataset_path='/opt/npu/deepFM/test_criteo_data/' \
    --checkpoint_path='./checkpint/deepfm-10_41258.ckpt' 


