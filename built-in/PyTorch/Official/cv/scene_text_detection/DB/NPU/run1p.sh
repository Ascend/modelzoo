export PYTHONPATH=./:$PYTHONPATH
export TASK_QUEUE_ENABLE=0
export DYNAMIC_OP="ADD"
python3.7 train.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
        --resume path-to-model-directory/MLT-Pretrain-ResNet50 \
        --seed=515 \
        --device_list "0"
