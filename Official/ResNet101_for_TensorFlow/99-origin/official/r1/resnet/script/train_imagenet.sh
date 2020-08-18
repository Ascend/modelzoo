
export PYTHONPATH=$PYTHONPATH:../../../
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 # seeeting your own gpu

NUM_GPUS=6
DTYPE=fp16
BATCH_SIZE_BASE=128
nohup python imagenet_main.py  \
--resnet_size 101 \
--dtype $DTYPE \
--batch_size $((BATCH_SIZE_BASE*$NUM_GPUS)) \
--train_epochs 90 \
--epochs_between_evals 50 \
--max_train_steps 1000 \
--data_dir /home/hiscv/dataset/imagenet_TF_record/ \
--model_dir training/test \
--hooks ExamplesPerSecondHook \
--num_gpus $NUM_GPUS > test.log &




