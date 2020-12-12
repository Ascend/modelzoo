#!/bin/bash

export install_path_atc=/usr/local/Ascend
export install_path_acllib=/usr/local/Ascend
export install_path_toolkit=/usr/local/Ascend
export install_path_opp=/usr/local/Ascend
export driver_path=/usr/local/Ascend
export ASCEND_OPP_PATH=${install_path_opp}/opp
export PATH=/usr/local/python3.7.5/bin:${install_path_atc}/atc/ccec_compiler/bin:${install_path_atc}/atc/bin:$PATH
export PYTHONPATH=${install_path_atc}/atc/python/site-packages/:${install_path_atc}/atc/python/site-packages/auto_tune.egg:${install_path_atc}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path_acllib}/acllib/lib64:${install_path_atc}/atc/lib64:${install_path_toolkit}/toolkit/lib64:${driver_path}/add-ons:$LD_LIBRARY_PATH

batchSize=1
soc_version=Ascend310
model=Bert_tnews
for para in $*
do
    if [[ $para == --batchSize* ]];then
        batchSize=`echo ${para#*=}`
	elif [[ $para == --soc_version* ]];then
		soc_version=`echo ${para#*=}`
	elif [[ $para == --model* ]];then
		model=`echo ${para#*=}`
	fi
done

cur_dir=`pwd`
out_path=$cur_dir

echo "start convert batchSize $batchSize om model"
atc --model=./${model}.pb --framework=3 --output=$out_path/"${model}_batch_$batchSize""_INT32_output_FP32" --input_format=NCHW --soc_version=$soc_version --input_shape="input_ids:$batchSize,128;input_mask:$batchSize,128;segment_ids:$batchSize,128" -precision_mode=force_fp16

