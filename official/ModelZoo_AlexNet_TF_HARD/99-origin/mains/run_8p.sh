#source /home/hiscv02/ngc/launch.sh

export TF_ENABLE_AUTO_MIXED_PRECISION=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

horovodrun -np 8 -H localhost:8 python mains/train.py --config_file alexnet_8p

