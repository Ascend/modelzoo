source ./env_b031.sh
source ./env_new.sh
export PYTHONPATH=./:$PYTHONPATH
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
/usr/local/Ascend/driver/tools/msnpureport -d 0 -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

cp disable.conf /home/disable.txt

export DYNAMIC_COMPILE_ENABLE=1
export EXPERIMENTAL_DYNAMIC_PARTITION=1
export TASK_QUEUE_ENABLE=1

export DISABLE_DYNAMIC_PATH=/home/disable.txt

export DYNAMIC_LOG_ENABLE=1
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/

export TRI_COMBINED_ENABLE=1
export COMBINED_ENABLE=1

export DYNAMIC_OP="ADD"



python3.7 tools/train_net.py \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
        --eval-only \
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        MODEL.DEVICE npu:0 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.WEIGHTS ./output/model_final.pth
