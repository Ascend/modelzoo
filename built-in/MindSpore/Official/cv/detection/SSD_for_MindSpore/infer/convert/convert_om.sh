#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 INPUT_AIR_PATH OUTPUT_OM_PATH_NAME"
  echo "Example: "
  echo "         bash convert_om.sh ./models/ssd-500_458_on_coco.air ./models/ssd-500_458_on_coco"

  exit 255
fi

input_air_path=$1
output_om_path=$2
aipp_cfg=$3

export ASCEND_ATC_PATH=/usr/local/Ascend/atc/bin/atc
export LD_LIBRARY_PATH=/usr/local/Ascend/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/atc/ccec_compiler/bin:/usr/local/Ascend/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/atc/python/site-packages:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/atc/python/site-packages/schedule_search.egg
export ASCEND_OPP_PATH=/usr/local/Ascend/opp

export ASCEND_SLOG_PRINT_TO_STDOUT=1

echo "Input AIR file path: ${input_air_path}"
echo "Output OM file path: ${output_om_path}"

atc --input_format=NCHW \
--framework=1 \
--model=${input_air_path} \
--output=${output_om_path} \
--soc_version=Ascend310 \
--disable_reuse_memory=0 \
--insert_op_conf=${aipp_cfg} \
--precision_mode=allow_fp32_to_fp16  \
--op_select_implmode=high_precision