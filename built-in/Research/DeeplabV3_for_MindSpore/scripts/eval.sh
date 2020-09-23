#!/bin/bash

export DEVICE_ID=3
export SLOG_PRINT_TO_STDOUT=0
export PYTHONPATH=/PATH/TO/MODEL_ZOO_CODE

python $PYTHONPATH/eval_seg_multiscale.py --data_root=/PATH/TO/DATA  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --batch_size=32  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --scales=1    \
                    --flip=False   \
                    --freeze_bn=False  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL \

