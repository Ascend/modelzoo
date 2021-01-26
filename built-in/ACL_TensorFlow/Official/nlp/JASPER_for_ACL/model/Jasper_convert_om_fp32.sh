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
for para in $*
do
    if [[ $para == --batchSize* ]];then
        batchSize=`echo ${para#*=}`
	elif [[ $para == --soc_version* ]];then
		soc_version=`echo ${para#*=}`
	fi
done

cur_dir=`pwd`
out_path=$cur_dir

echo "start convert batchSize $batchSize om model"
atc --model=./jasper_infer_float32.pb --framework=3 --output=$out_path/"jasper_b$batchSize""_output_FP32" --soc_version=$soc_version --input_shape="input_shape:$batchSize,1,2336,64;input_reshape:$batchSize,1168" 



