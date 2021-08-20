#!/bin/bash
export install_path=/usr/local/Ascend
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

export SLOG_PRINT_TO_STDOUT=1

onnx_path=$1
om_path=$2

# bash convert_om.sh ./wdl.onnx ./wdl

echo "Input ONNX file path: ${onnx_path}"
echo "Output OM file path: ${om_path}"

atc --framework=5 --model="${onnx_path}" \
    --input_shape="actual_input_1:1,39" \
    --output="${om_path}" \
    --precision_mode="allow_mix_precision" \
    --log=error \
    --soc_version=Ascend310
