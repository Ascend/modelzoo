#!/bin/bash
#set -x
cur_dir=`pwd`
benchmark_dir=$cur_dir/../Benchmark/out
om_name=$cur_dir/../shadownet_tf_64batch.om

#start offline inference
$benchmark_dir/benchmark --model $om_name --input $cur_dir/img_bin/ --output $cur_dir/output_fp32

#post process
python3 tools/post_precess.py
