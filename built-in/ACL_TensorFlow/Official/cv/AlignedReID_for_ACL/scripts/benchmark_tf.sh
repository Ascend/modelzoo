#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../AlignedReID_100batch.om
batchsize=100
model_name=AlignedReID
output_dir='results'
rm -rf $cur_dir/$output_dir/*

#Inference and postprocess
python3 calc_cmc.py
