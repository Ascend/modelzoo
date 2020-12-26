#!/bin/bash

inferRet_folder="../model1_dev_0_chn_0_results/jasper/"
real_file="../datasets/librivox-dev-clean.csv"
for para in $*
do
        if [[ $para == --inferRet_folder* ]];then
                inferRet_folder=`echo ${para#*=}`
        elif [[ $para == --real_file* ]];then
                real_file=`echo ${para#*=}`
	fi
done

python3 jasper_post_process_fp32.py $inferRet_folder $real_file

