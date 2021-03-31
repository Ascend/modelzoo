unset PROFILING_DIR
unset PROFILING_MODE
unset PROFILING_OPTIONS
export PROFILING_DIR=/var/log/npu/profiling/container/0
mkdir -p /root/ide_daemon
cd /root/ide_daemon
rm -f `ls /home/HwHiAiUser/ide_daemon/ide*`
cp /home/HwHiAiUser/ide_daemon/ide* /root/ide_daemon
pkill ada
sleep 2
/usr/local/Ascend/driver/tools/ada
sleep 1 
mkdir -p /var/log/npu/profiling/container/0
cd -
cd /var/log/npu/profiling
rm -rf JOB*
chown -R HwHiAiUser:HwHiAiUser container
cd -

export JOB_ID=123456789
export PROFILING_MODE=true
export AICPU_PROFILING_MODE=true
export PROFILING_OPTIONS=training_trace:task_trace

export FP_POINT=megatron/GatherV2_1
export BP_POINT=loss_scale/gradients_4/loss_scale/megatron/gpt2_block_0/TransformerLayer/layer_normalization/sub_grad/Sum

