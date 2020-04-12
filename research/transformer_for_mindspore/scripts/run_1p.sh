source ./scripts/env_single.sh

rm -rf run_single
cp ./transformer ./run_single -rf
cd run_single

python train_main.py --data_path "../data/tfrecord/" --train_epochs 7 --batch_size 96 --checkpoint_path "./model_dir/" | tee log_run_single

cd ..
