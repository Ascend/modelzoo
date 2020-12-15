source ./env_b031.sh
export PYTHONPATH=./:$PYTHONPATH
export SLOG_PRINT_TO_STDOUT=0
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(0)[error]\" --device 4"
su HwHiAiUser -c "adc --host 0.0.0.0:22118 --log \"SetLogLevel(2)[disable]\" "

nohup python3.7 tools/train_net.py \
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
        --debug-mode 0\
        AMP 1\
        OPT_LEVEL O2 \
        LOSS_SCALE_VALUE 64 \
        MODEL.DEVICE npu:5 \
        SOLVER.IMS_PER_BATCH 2 \
        SOLVER.MAX_ITER 82000 \
        SEED 1234 \
        MODEL.RPN.NMS_THRESH 0.8 \
        MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO 2 \
        MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO 2 \
        DATALOADER.NUM_WORKERS 4 \
        SOLVER.BASE_LR 0.0025 &
