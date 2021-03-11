#!/bin/bash
air_path=$1
om_path=$2

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_SLOG_PRINT_TO_STDOUT=1

echo "Input AIR file path: ${air_path}"
echo "Output OM file path: ${om_path}"

atc --input_format=NCHW \
    --framework=1 --model="${air_path}" \
    --input_shape="x:1, 3, 768, 1280" \
    --output="${om_path}" \
    --insert_op_conf=./aipp.cfg \
    --precision_mode=allow_fp32_to_fp16 \
    --soc_version=Ascend310
