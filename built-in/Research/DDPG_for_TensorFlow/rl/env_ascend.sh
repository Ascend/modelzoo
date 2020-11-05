#export SUMO_HOME="/home/sumo-master"
#export PATH="/home/sumo-master/bin:$PATH"


export ASCEND_OPP_PATH=/usr/local/Ascend/opp


export DEVICE_ID=0
export DEVICE_INDEX=0
export PRINT_MODEL=1
#export DUMP_GRAPH_LEVEL=1
#export DUMP_GE_GRAPH=1


export RANK_TABLE_FILE=./hccl.json
export RANK_ID=0
export RANK_SIZE=1
export RANK_INDEX=0
export JOB_ID=10087

export SOC_VERSION=Ascend910

echo $PYTHONPATH

#bash rm_logs.sh

#python3 tf_train.py
xt_main -f examples/default_cases/ddpg_pendulum.yaml
