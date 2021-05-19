#!/bin/bash

rm -rf resnext50.onnx
python3.7 resnext50_pth2onnx.py resnext50_32x4d-7cdf4587.pth resnext50.onnx
source env.sh
rm -rf resnext50_bs1.om resnext50_bs16.om
atc --framework=5 --model=./resnext50.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=resnext50_bs1 --log=debug --soc_version=Ascend310
atc --framework=5 --model=./resnext50.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=resnext50_bs16 --log=debug --soc_version=Ascend310
if [ -f "resnext50_bs1.om" ] && [ -f "resnext50_bs16.om" ]; then
    echo "success"
else
    echo "fail!"
fi