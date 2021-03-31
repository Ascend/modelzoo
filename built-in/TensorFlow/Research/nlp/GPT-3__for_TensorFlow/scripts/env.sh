#!/bin/bash

rm -rf /var/log/npu/slog/host-0/*
rm -rf /var/log/npu/slog/device*

export PYTHONPATH=/home/models/ModelZoo_WideDeep_TF:/usr/local/Ascend/ops/op_impl/built-in/ai_core/tbe/:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/add-ons
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/fwkacllib/python/site-packages/auto_tune.egg:/usr/local/Ascend/fwkacllib/python/site-packages/schedule_search.egg
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
