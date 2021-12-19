#!/bin/bash
export install_path=/usr/local/Ascend
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/hdf/hdf5/lib:${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:$PYTHONPATH
export PYTHONPATH=${install_path}/tfplugin/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export JOB_ID=10087
export ASCEND_DEVICE_ID=3
export EXPERIMENTAL_DYNAMIC_PARTITION=1
python3 train-tf.keras-npu.py
