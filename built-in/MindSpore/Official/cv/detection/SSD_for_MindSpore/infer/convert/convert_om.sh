#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 ASCEND_TOOLKIT_INSTALL_PATH INPUT_AIR_PATH OUTPUT_OM_PATH_NAME"
  echo "Example: "
  echo "         bash convert_om.sh /usr/local/Ascend/ascend-toolkit/ ./models/ssd-500_458_on_coco.air ./models/ssd-500_458_on_coco"

  exit 255
fi

ascend_toolkit_install_path=$1
input_air_path=$2
output_om_path=$3
aipp_cfg=$4

export install_path=$ascend_toolkit_install_path

export ASCEND_ATC_PATH=${install_path}/latest/atc
export LD_LIBRARY_PATH=${install_path}/latest/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:${install_path}/latest/atc/ccec_compiler/bin:${install_path}/latest/atc/bin:${install_path}/latest/toolkit/bin:$PATH
export PYTHONPATH=${install_path}/latest/atc/python/site-packages:${install_path}/latest/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/latest/atc/python/site-packages/schedule_search.egg
export ASCEND_OPP_PATH=${install_path}/latest/opp

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