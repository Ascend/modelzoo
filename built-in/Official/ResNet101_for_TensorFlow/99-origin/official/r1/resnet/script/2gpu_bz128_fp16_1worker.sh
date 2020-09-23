
export PYTHONPATH=$PYTHONPATH:../../../
export CUDA_VISIBLE_DEVICES=2,3 # seeeting your own gpu

NUM_GPUS=2
DTYPE=fp16
BATCH_SIZE_BASE=128

nohup python imagenet_main.py  \
--resnet_size 101 \
--dtype $DTYPE \
--batch_size $((BATCH_SIZE_BASE*$NUM_GPUS)) \
--train_epochs 90 \
--data_dir /home/hiscv/dataset/imagenet_TF_record/ \
--model_dir training/$(basename $0 .sh) \
--hooks LoggingTensorHook,ExamplesPerSecondHook,loggingmetrichook,stepcounterhook \
--num_gpus $NUM_GPUS > $(basename $0 .sh) &
