export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest/
export LD_LIBRARY_PATH=/usr/local/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:$ASCEND_HOME/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$PATH:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/toolkit/tools/ide_daemon/bin/
export ASCEND_OPP_PATH=$ASCEND_HOME/opp/
export OPTION_EXEC_EXTERN_PLUGIN_PATH=$ASCEND_HOME/fwkacllib/lib64/plugin/opskernel/libfe.so:$ASCEND_HOME/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:$ASCEND_HOME/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages/:$ASCEND_HOME/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:$ASCEND_HOME/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH

source env_new.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1