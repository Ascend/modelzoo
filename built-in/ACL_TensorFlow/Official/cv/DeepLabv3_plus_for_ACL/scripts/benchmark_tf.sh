#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../deeplabv3_plus_tf_1batch.om
batchsize=1
model_name=deeplabv3_plus
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
python3 $cur_dir/postprocess.py $cur_dir/PascalVoc2012/ $cur_dir/$output_dir/$model_name
