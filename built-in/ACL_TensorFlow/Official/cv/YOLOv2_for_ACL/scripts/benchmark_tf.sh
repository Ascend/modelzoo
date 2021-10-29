#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../yolov2_tf_1batch.om
batchsize=1
model_name=yolov2
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#start offline inference
$benchmark_dir/benchmark --om $om_name --dataDir $cur_dir/input_bins/ --modelType $model_name --outDir $cur_dir/$output_dir --batchSize $batchsize --imgType bin --useDvpp 0

#post process
cd yolov2_postprocess
python3 yolov2_postprocess.py $cur_dir/$output_dir/$model_name ../VOC2007/JPEGImages/
python3 pascalvoc.py --detfolder ./detections_npu/ -np
