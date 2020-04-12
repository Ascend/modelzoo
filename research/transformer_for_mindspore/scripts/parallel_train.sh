

python train_main.py \
  --data_path "../../data/tfrecord/" \
  --train_epochs 52 \
  --batch_size 96 \
  --checkpoint_path "./model_dir/" | tee log_parallel
