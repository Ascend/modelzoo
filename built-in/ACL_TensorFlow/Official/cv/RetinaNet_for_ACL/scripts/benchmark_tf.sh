#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../retinanet_tf_1batch.om
batchsize=1
model_name=retinanet
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
cd retinanet_postprocess
python3 retinaPostprocess.py $cur_dir/$output_dir/$model_name ./detections/
python3 pascalvoc.py -np
