
export PYTHONPATH=$PYTHONPATH:/home/hiscv/w00558981/resnet101_gpu
export CUDA_VISIBLE_DEVICES=2

python cifar10_main.py \
--data_dir /home/hiscv/dataset/cifar10/cifar-10-batches-bin/ \
--model_dir cifar10_training \
--num_gpus 1
