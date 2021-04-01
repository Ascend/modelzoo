unset PROFILING_DIR
unset PROFILING_MODE
unset PROFILING_OPTIONS
export PROFILING_DIR=/var/log/npu/profiling/container/0
mkdir -p /root/ide_daemon
cd /root/ide_daemon
rm -f `ls /home/HwHiAiUser/ide_daemon/`
cp -rf /home/HwHiAiUser/ide_daemon/* /root/ide_daemon
#pkill IDE-daemon-host
pkill ada
#/usr/local/HiAI/driver/tools/IDE-daemon-host
/usr/local/HiAI/driver/tools/ada
sleep 5
mkdir -p /var/log/npu/profiling/container/0
#export JOB_ID=1558618389
export PROFILING_MODE=true
export AICPU_PROFILING_MODE=true
#export PROFILING_DIR=/var/log/npu/profiling/container/$DEVICE_ID
export PROFILING_OPTIONS=training_trace:task_trace
cd -
cd /var/log/npu/profiling
rm -rf JOB*
chown -R HwHiAiUser:HwHiAiUser container
cd -

#export FP_POINT=fp32_vars/conv2d/Conv2Dfp32_vars/BatchNorm/FusedBatchNormV3_Reduce
export FP_POINT=dense/Relu
export BP_POINT=train/update_embedding_item/embeddings/ResourceApplyAdam
#export BP_POINT=loss_scale/gradients/AddN_70
