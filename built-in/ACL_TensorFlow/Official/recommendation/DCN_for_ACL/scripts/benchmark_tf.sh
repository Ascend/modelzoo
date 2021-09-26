#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../dcn_tf_4000batch.om

#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/input_x/ --output $cur_dir/output

#post process
python3 eval.py ./labels/ $cur_dir/output
