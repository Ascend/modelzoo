pwd=`pwd`
dir_name=`dirname $pwd`

export SOC_VERSION=Ascend910
export JOB_ID=10087
export DEVICE_ID=$1
export RANK_TABLE_FILE=${pwd}/rank_table_file.json
export RANK_ID=$1
export RANK_SIZE=8

export PYTHONPATH=${dir_name}:$PYTHONPATH
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_TASK_INDEX=0
export TF_CPP_MIN_LOG_LEVEL=3

python3 ${pwd}/run_example.py ${pwd}/nas/darts_cnn/darts_tf.yml

