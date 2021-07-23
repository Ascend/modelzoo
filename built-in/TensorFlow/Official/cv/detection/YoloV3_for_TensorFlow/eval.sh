
#export CUDA_VISIBLE_DEVICES=''
#export CUDA_VISIBLE_DEVICES=7



# setting main path
MAIN_PATH=$(dirname $(readlink -f $0))

# set env
export DDK_VERSION_FLAG=1.60.T49.0.B201
export NEW_GE_FE_ID=1
export GE_AICPU_FLAG=1
export SOC_VERSION=Ascend910

export JOB_ID=10087
export FUSION_TENSOR_SIZE=1000000000

for((RANK_ID=0;RANK_ID<8;RANK_ID++));
do

export RANK_ID=$RANK_ID
export RANK_SIZE=1
export DEVICE_ID=$RANK_ID
export DEVICE_INDEX=$RANK_ID

RESTORE_PATH=./training/t1/D$RANK_ID/training/

nohup python3.7 eval.py \
--save_json True \
--score_thresh 0.0001 \
--nms_thresh 0.55 \
--max_boxes 100 \
--restore_path $RESTORE_PATH \
--max_test 10000 \
--save_json_path eval_res_D$RANK_ID.json > eval_$RANK_ID.out &


done


