source ./env_b031.sh
export PYTHONPATH=./:$PYTHONPATH
export SLOG_PRINT_TO_STDOUT=0
export GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export PTCOPY_ENABLE=1
/usr/local/Ascend/driver/tools/msnpureport -d 4 -g error
/usr/local/Ascend/driver/tools/msnpureport -e disable

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        MODEL.DEVICE npu:5 \
        SOLVER.IMS_PER_BATCH 8 \
        SOLVER.MAX_ITER 90000 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 4 \
        SOLVER.BASE_LR 0.0025 &
