#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../yolov4_tf_1batch.om
batchsize=1
model_name=yolov4
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
cd yolov4_postprocess
python3 yolov4_postprocess.py $cur_dir/$output_dir/$model_name ../val2017/
python3 pascalvoc.py --detfolder ./detections_npu/ -np
