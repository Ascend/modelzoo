#!/bin/bash
export install_path=/usr/local/Ascend
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp

export SLOG_PRINT_TO_STDOUT=1

/usr/local/Ascend/atc/bin/atc \
	--model=$1 \
	--framework=5 \
	--output=$2 \
	--input_format=NCHW \
	--input_shape="actual_input_1:1,3,304,304" \
	--enable_small_channel=1 \
	--log=error \
	--soc_version=Ascend310 \
	--insert_op_conf=densenet121_pt_aipp.config
