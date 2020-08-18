
export PYTHONPATH=$PYTHONPATH:../../../
export CUDA_VISIBLE_DEVICES=2,3 # seeeting your own gpu

NUM_GPUS=2
DTYPE=fp32
BATCH_SIZE_BASE=128

ALL_REDUCE_ALG=nccl # nccl / ring as well

nohup python imagenet_main.py  \
--resnet_size 101 \
--dtype $DTYPE \
--batch_size $((BATCH_SIZE_BASE*$NUM_GPUS)) \
--train_epochs 90 \
--data_dir /home/hiscv/dataset/imagenet_TF_record/ \
--model_dir training/$(basename $0 .sh) \
--tf_gpu_thread_mode gpu_private \
--intra_op_parallelism_threads 1 \
--datasets_num_private_threads $((4*$NUM_GPUS)) \
--distribution_strategy multi_worker_mirrored \
--all_reduce_alg $ALL_REDUCE_ALG \
--hooks LoggingTensorHook,ExamplesPerSecondHook,ProfilerHook,loggingmetrichook,stepcounterhook \
--num_gpus $NUM_GPUS > $(basename $0 .sh).log &
