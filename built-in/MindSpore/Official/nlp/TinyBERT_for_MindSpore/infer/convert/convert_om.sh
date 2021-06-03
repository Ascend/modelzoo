#!/bin/bash
air_path=$1
om_path=$2

echo "Input AIR file path: ${air_path}"
echo "Output OM file path: ${om_path}"

atc --framework=1 --model="${air_path}" \
    --output="${om_path}" \
    --soc_version=Ascend310 \
    --op_select_implmode="high_precision"
