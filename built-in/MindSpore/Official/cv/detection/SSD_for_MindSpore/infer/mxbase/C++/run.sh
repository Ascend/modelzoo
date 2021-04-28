#!/bin/bash

export ASCEND_HOME=/usr/local/Ascend
export ARCH_PATTERN=x86_64-linux
export ASCEND_VERSION=ascend-toolkit/latest

MXBASE_CODE_DIR=/data/sam/codes/mindxsdk-mxbase
OM_FILE=/data/pretrained_models/ms/mobilenet_v1/ckpt_0/mobilenetv1-90_625_2.om

export MXSDK_OPENSOURCE_DIR=${MXBASE_CODE_DIR}/opensource/dist

export LD_LIBRARY_PATH=${MXBASE_CODE_DIR}/dist/opensource/lib:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64

cd $MXBASE_CODE_DIR
rm -rf dist
mkdir dist
cd dist
cmake ..
make -j
make install

cd ${MXBASE_CODE_DIR}/samples/C++/
cp ${MXBASE_CODE_DIR}/dist/samples/C++/ssd_mobilenet_v1_fpn .

./ssd_mobilenet_v1_fpn ${OM_FILE} ./test.jpg