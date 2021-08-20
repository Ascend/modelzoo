#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../pixellink_tf_1batch.om
batchsize=1
model_name=pixellink
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
mkdir $cur_dir/txt
python3 postprocess.py $cur_dir/input_bins/ $cur_dir/txt
#Caculate Hmean
python3 evaluation/script.py -g=evaluation/gt.zip -s=$cur_dir/txt/result.zip
