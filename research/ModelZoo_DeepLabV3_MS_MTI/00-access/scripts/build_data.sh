#!/bin/bash

export DEVICE_ID=7
python /PATH/TO/MODEL_ZOO_CODE/data/build_seg_data.py  --data_root=/PATH/TO/DATA_ROOT  \
                    --data_lst=/PATH/TO/DATA_lst.txt  \
                    --dst_path=/PATH/TO/MINDRECORED_NAME.mindrecord  \
                    --num_shards=8  \
                    --shuffle=True