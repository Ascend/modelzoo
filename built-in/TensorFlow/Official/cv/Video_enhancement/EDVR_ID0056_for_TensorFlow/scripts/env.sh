# !/bin/bash

export CUSTOM_OP_LIB_PATH=/usr/local/Ascend/fwkacllib/ops/framework/built-in/tensorflow/

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:${PYTHONPATH}

export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/toolkit/bin:/usr/local/Ascend/fwkacllib/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp