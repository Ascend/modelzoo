export SUMO_HOME="/home/ModelZoo_PPO_TF_NOAH/sumo-master"
export PATH="$SUMO_HOME/bin:$PATH"


# set env


export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
export DEVICE_ID=0
export DEVICE_INDEX=0

# export RANK_TABLE_FILE=./hccl.json
export RANK_TABLE_FILE=/home/ModelZoo_PPO_TF_NOAH/hccl.json
export RANK_ID=0
export RANK_SIZE=1
export RANK_INDEX=0
export JOB_ID=10087

export SOC_VERSION=Ascend910

echo $PYTHONPATH

