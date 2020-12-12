export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/local/OpenBLAS/lib/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/te:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/topi:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/hccl:/usr/local/Ascend/ascend-toolkit/latest/tfplugin/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/python/site-packages/te:$currentDir
export PATH=$PATH:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp


export SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1