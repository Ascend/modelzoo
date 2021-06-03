#!/bin/bash
#env variables
export install_path=/usr/local/Ascend
export LD_LIBRARY_PATH=${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:$PYTHONPATH
export PYTHONPATH=${install_path}/tfplugin/python/site-packages:$PYTHONPATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}
export ASCEND_DEVICE_ID=0