#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/fwkacllib/python/site-packages:/usr/local/Ascend/tfplugin/python/site-packages
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/atc/python/site-packages:/usr/local/Ascend/atc/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/atc/python/site-packages/schedule_search.egg:$PYTHONPATH

python3.7 infer_from_pb.py --batchsize=8 --model_path=./pb_model_tf/densenet_tf_910.pb --image_path=./image-1024/ --label_file=./val_lable.txt
