
export TF_ENABLE_AUTO_MIXED_PRECISION=1
export CUDA_VISIBLE_DEVICES=0
horovodrun -np 8 -H localhost:8 python mains/train.py --config_file alexnet_1p
