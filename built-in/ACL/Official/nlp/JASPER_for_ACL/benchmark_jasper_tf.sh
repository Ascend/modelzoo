#!/bin/bash
#set -x
cur_dir=`pwd`
log_dir=$cur_dir/testlog/


if [  ! -d "$log_dir" ];then
    mkdir -p $log_dir
fi

preprocess_fp32()
{
    cd $cur_dir/pre_treatment/
    chmod +x prerun_jasper_infer_input_fp32.sh 
    ./prerun_jasper_infer_input_fp32.sh | tee $testcase_dir/preprocess.txt
    cd $cur_dir
}

infer_test()
{
    cd $cur_dir/output
    chmod +x benchmark
    echo "./benchmark $json" | tee $testcase_dir/infer.txt
    ./benchmark $json | tee $testcase_dir/infer.txt
}

collect_result_fp32()
{
    cd $cur_dir/post_treatment
	chmod +x jasper_accuracy_calc_fp32.sh
    ./jasper_accuracy_calc_fp32.sh | tee $testcase_dir/postprocess.txt
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
    ./benchmark.sh --preprocess=1 --json=jasper_syn_inference_01.json --type=fp32"
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

echo "======================preprocess FP32========================"
if [[ $preprocess == 1 ]];then
    rm -rf $cur_dir/datasets/jasper/input_0
    rm -rf $cur_dir/datasets/jasper/input_reshape
    preprocess_fp32
fi
echo "======================infer test========================"
infer_test
echo "======================collect results==================="
collect_result_fp32
echo "======================end==============================="


