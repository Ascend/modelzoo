#!/bin/bash
rm -rf Onnxgraph
rm -rf Partition
rm -rf OptimizeSubGraph
rm -rf Aicpu_Optimized
rm *txt
rm -rf result_$RANK_ID
rm -rf model_dir/events*
sed -i 's/10/0/g' model_dir/checkpoint

export RANK_ID=$1
export RANK_SIZE=$2

export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID
export RANK_TABLE_FILE=rank_table.json

export JOB_ID=10087
export FUSION_TENSOR_SIZE=1000000000


#sleep 5
python3 $3/imagenet_main.py \
--resnet_size=101 \
--batch_size=128 \
--num_gpus=1 \
--cosine_lr=True \
--dtype=fp16 \
--label_smoothing=0.1 \
--loss_scale=512 \
--train_epochs=1 \
--epochs_between_evals=1 \
--hooks=ExamplesPerSecondHook,loggingtensorhook,loggingmetrichook \
--data_dir=$3/test_data \
--model_dir=./model_dir \
--test_mode=True \
--max_train_steps=10 \
--iterations_per_loop=1 \
--random_seed=123

mkdir graph
mv *.txt graph
mv *.pbtxt graph

#rm $3/test_data/tmp*