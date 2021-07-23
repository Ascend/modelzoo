#!/bin/bash
# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cur_dir=$(cd "$(dirname "$0")" || exit; pwd)
soc_version=$1

# Source environment variable
echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - I - [TfPose]: Start to source environment variable"
if [ "A${soc_version}" == "A" ];then
  source "${cur_dir}"/env/Ascend310_env.ini
elif [ "A${soc_version}" == "AAscend310" ];then
  source "${cur_dir}"/env/Ascend310_env.ini
else
  echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - E - [TfPose]: Unsupported device type: ${soc_version}"
  exit 1
fi

# Install pip dependency
echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - I - [TfPose]: Start to install dependency packages"
python_flag=$(which python3)
if [ "A${python_flag}" == "A" ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - E - [TfPose]: Python3 is not installed"
  exit 1
fi

pip_flag=$(which pip3)
if [ "A${pip_flag}" == "A" ]; then
  echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - E - [TfPose]: Pip3 is not installed"
  exit 1
fi

pip3 install -r "${cur_dir}"/requirements.txt > /dev/null 2>&1

if [ $? -ne 0 ];then
  echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - E - [TfPose]: Python dependency packages install failed"
  exit 1
fi

cd "${cur_dir}"/libs/pafprocess || exit
apt install swig >/dev/null 2>&1 && swig -python -c++ pafprocess.i >/dev/null 2>&1 && python3 setup.py build_ext --inplace >/dev/null 2>&1

if [ $? -ne 0 ];then
  echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - E - [TfPose]: Pafprocess package install failed"
  exit 1
else
  cd "${cur_dir}" || exit
fi

# Preprocess
echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - I - [TfPose]: Start to do the preprocess"
if [ -d "${cur_dir}/input" ]; then
  rm -rf "${cur_dir}"/input
fi

cd "${cur_dir}"/libs || exit

python3 preprocess.py \
    --resize 656x368 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir "${cur_dir}"/dataset/coco/ \
    --output-dir "${cur_dir}"/input/

cd "${cur_dir}" || exit

# ATC MODEL
echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - I - [TfPose]: Start to generate npu model"
atc --framework=3 \
    --model="${cur_dir}"/models/OpenPose_for_TensorFlow_BatchSize_1.pb \
    --output="${cur_dir}"/models/OpenPose_for_TensorFlow_BatchSize_1 \
    --soc_version=Ascend310 \
    --input_shape="image:1,368,656,3"

# RUN ACL
echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - I - [TfPose]: Start to run npu predict"
if [ -d "${cur_dir}/output" ]; then
  rm -rf "${cur_dir}"/output
fi

./xacl_fmk -m "${cur_dir}"/models/OpenPose_for_TensorFlow_BatchSize_1.om \
    -o "${cur_dir}"/output/openpose \
    -i "${cur_dir}"/input \
    -b 1

# Postprocess
cd "${cur_dir}"/libs || exit

echo "$(date +'%Y-%m-%d %H:%M:%S,%3N') - I - [TfPose]: Start to do the postprocess"
python3 postprocess.py \
    --resize 656x368 \
    --resize-out-ratio 8.0 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir "${cur_dir}"/dataset/coco/ \
    --data-idx 100 \
    --input-dir "${cur_dir}"/input \
    --output-dir "${cur_dir}"/output

cd "${cur_dir}" || exit
