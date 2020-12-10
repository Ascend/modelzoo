export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/te:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/topi:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/hccl:/usr/local/Ascend/ascend-toolkit/latest/tfplugin/python/site-packages:$currentDir
export PATH=$PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/

export SLOG_PRINT_TO_STDOUT=0
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 7"

export TASK_QUEUE_ENABLE=0
taskset -c 111-150 python3 densenet121_1p_main.py \
	--workers 40 \
	--arch densenet121 \
	--npu 7 \
	--lr 0.1 \
	--momentum 0.9 \
	--amp \
	--batch-size 256 \
	--epoch 90 \
	--evaluate \
	--resume checkpoint.pth.tar \
	--data /opt/npu/dataset/imagenet