#!/bin/bash

export DEVICE_ID=7
export SLOG_PRINT_TO_STDOUT=0
python /PATH/TO/MODEL_ZOO_CODE/infer_seg.py  --image_lst=/PATH/TO/MODEL_ZOO_CODE/img/infer/image.txt  \
                    --dst_dir=/PATH/TO/MODEL_ZOO_CODE/img/infer/mask/  \
                    --crop_size=513  \
                    --ignore_label=255  \
                    --num_classes=21  \
                    --model=deeplab_v3_s16  \
                    --freeze_bn=False  \
                    --ckpt_path=/PATH/TO/PRETRAIN_MODEL  \
                    --palette_path=/PATH/TO/palette.pkl


