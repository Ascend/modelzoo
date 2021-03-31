# set env
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:/usr/local/Ascend/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin
export ASCEND_OPP_PATH=/usr/local/Ascend/opp

# user env
export JOB_ID=NPU20210126
export RANK_SIZE=1
export RANK_ID=0 
export ASCEND_DEVICE_ID=0

# debug env
export ASCEND_GLOBAL_LOG_LEVEL=3
#export SLOG_PRINT_TO_STDOUT=0
#export DUMP_GE_GRAPH=1
#export DUMP_GRAPH_LEVEL=3

currentDir=$(cd "$(dirname "$0")"; pwd)
cd ${currentDir}

data_path=$1

start_time=`date +%Y%m%d%H%M%S`
echo "=== start at ${start_time} ==="

python3 unet3d_pb_inference.py ${data_path}

end_time=`date +%Y%m%d%H%M%S`
echo "=== finish at ${end_time} ==="

