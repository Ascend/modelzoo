export DDK_VERSION_FLAG=1.72.T2.0.B020

export WHICH_OP=GEOP
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1

export RANK_TABLE_FILE=./ad_hccl.json

export DEVICE_ID=1
export PRINT_MODEL=1

export RANK_ID=0
export RANK_SIZE=1
export JOB_ID=10087

export SOC_VERSION=Ascend910

export PROFILING_MODE=false

echo $PYTHONPATH

# start train
python3 xt/main.py -f examples/ma_cases/qmix.yaml -t train
