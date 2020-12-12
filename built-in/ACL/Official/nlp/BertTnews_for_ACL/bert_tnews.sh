#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cur_dir=`pwd`
log_dir=$cur_dir/testlog/


if [  ! -d "$log_dir" ];then
    mkdir -p $log_dir
fi

preprocess()
{
    cd $cur_dir/pre_treatment/
    ./run_bert_tnews_infer_input.sh | tee $testcase_dir/preprocess.txt
    cd $cur_dir
}

infer_test()
{
    cd $cur_dir/output
    echo "cd $cur_dir/output"
    chmod +x bert_infer
    echo "./bert_infer $json" | tee $testcase_dir/infer.txt
    ./bert_infer $json | tee $testcase_dir/infer.txt
}

collect_result()
{
    cd $cur_dir/post_treatment
    ./bert_tnews_accuracy_calc.sh | tee $testcase_dir/postprocess.txt
    cp $cur_dir/output/predict_accuracy.txt $testcase_dir/
    echo "======================precision results==================="
    cat $testcase_dir/predict_accuracy.txt
    echo "======================performance results==================="
    cp $cur_dir/output/perform_static_dev_0_chn_0.txt $testcase_dir/
    cat $testcase_dir/perform_static_dev_0_chn_0.txt
}


###########################START#########################

if [[ $1 == --help || $1 == -h || $# -eq 0 ]];then 
    echo "usage: ./benchmark.sh <args>"
#    echo "example:
#    
#    Execute the following command when model file is converted:
#    ./benchmark.sh --preprocess=1"
#
    echo ""
    echo 'parameter explain:
    --preprocess           (0-no preprocess,1--preprocess, default is 0)
    -h/--help             Show help message
    Example:
    ./benchmark_tf.sh --preprocess=0 --json=inference_syn_bert_b1.json"
#    '
    exit 1
fi

preprocess=1
json=""

for para in $*
do
    if [[ $para == --preprocess* ]];then
        preprocess=`echo ${para#*=}`
    elif [[ $para == --json* ]];then
        json=`echo ${para#*=}`
    fi
done    

rm -rf $cur_dir/model1_dev_0_chn_0_results
rm -rf $cur_dir/output/perform_static_dev_0_chn_0.txt
rm -rf $cur_dir/output/predict_accuracy.txt

testcase_dir=$log_dir/$(date "+%Y%m%d%H%M%S")
mkdir -p $testcase_dir 

echo "======================preprocess========================"
if [[ $preprocess == 1 ]];then
    rm -rf $cur_dir/datasets/bert_tnews/*
    preprocess
fi
echo "======================infer test========================"
infer_test
echo "======================collect results==================="
collect_result
echo "======================end==============================="


