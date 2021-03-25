#!/bin/bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

datasets_folder="$CURRENT_DIR/../datasets"
output_folder="$CURRENT_DIR/../datasets/jasper"
for para in $*
do
        if [[ $para == --datasets_folder* ]];then
                datasets_folder=`echo ${para#*=}`
        elif [[ $para == --output_folder* ]];then
                output_folder=`echo ${para#*=}`
	fi
done

python3 jasper_prep_process_fp32.py $datasets_folder $output_folder

