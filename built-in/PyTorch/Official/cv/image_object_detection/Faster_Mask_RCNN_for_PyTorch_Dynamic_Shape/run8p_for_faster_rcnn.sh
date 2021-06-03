source ./env_b031.sh
source ./env_new.sh
export PYTHONPATH=./:$PYTHONPATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
export DYNAMIC_OP="ADD"


export DYNAMIC_COMPILE_ENABLE=1
export EXPERIMENTAL_DYNAMIC_PARTITION=1

export DISABLE_DYNAMIC_PATH=./disable.conf

export DYNAMIC_LOG_ENABLE=0
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/

export TRI_COMBINED_ENABLE=1
export COMBINED_ENABLE=1

export HCCL_WHITELIST_DISABLE=1

/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
        --device-ids 0 1 2 3 4 5 6 7 \
        --num-gpus 8\
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        SOLVER.IMS_PER_BATCH 64 \
        SOLVER.MAX_ITER 11250 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 8 \
        SOLVER.BASE_LR 0.02 &
