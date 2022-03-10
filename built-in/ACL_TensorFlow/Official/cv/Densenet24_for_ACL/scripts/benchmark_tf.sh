#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../model/densenet24_1batch.om
batchsize=1
model_name=densenet24
output_dir='results'
rm -rf $cur_dir../$output_dir/*

$benchmark_dir/benchmark --dataDir ../datasets/input_flair/,../datasets/input_t1/ --om $om_name --batchSize 1 --modelType $model_name --imgType bin --deviceId 0 --framework tensorflow --useDvpp 0

python3 $cur_dir/postprocess.py -m $cur_dir/../ori_images/npu/dense24_correction-4 -mn dense24 -nc True -r $cur_dir/../ori_images/BRATS2017/Brats17ValidationData/ --rp $cur_dir/../../results/$model_name/