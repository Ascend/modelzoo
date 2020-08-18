#!/bin/bash
dos2unix /path/to/modelzoo_yolo/scripts/env.sh

pretraind_root="/path/to/ckpt/"

for dv in 320 319 318 317 316 315 314 313
do
    mkdir /path/to/modelzoo_yolo/model_test/$dv
    cd /path/to/modelzoo_yolo/model_test/$dv
    if [ $dv == 320 ]
    then 
        ckpt="0-320_102400.ckpt"
        dv_id=0
    fi
    
    if [ $dv == 319 ]
    then 
        ckpt="0-319_102080.ckpt"
        dv_id=1
    fi

    if [ $dv == 318 ]
    then
        ckpt="0-318_101760.ckpt"
        dv_id=2
    fi

    if [ $dv == 317 ] 
    then
        ckpt="0-317_101440.ckpt"
        dv_id=3
    fi


    if [ $dv == 316 ]
    then
        ckpt="0-316_101120.ckpt"
        dv_id=4
    fi

    if [ $dv == 315 ]
    then
        ckpt="0-315_100800.ckpt"
        dv_id=5
    fi

    if [ $dv == 314 ]
    then
        ckpt="0-314_100480.ckpt"
        dv_id=6
    fi

    if [ $dv == 313 ]
    then
        ckpt="0-313_100160.ckpt"
        dv_id=7
    fi
    

 python /path/to/modelzoo_yolo/launch.py \
        --nproc_per_node=1 \
        --mode=test \
        --visible_devices=$dv_id \
        --server_id="xx.xxx.xxx.xxx" \
        --env_sh="/path/to/modelzoo_yolo/scripts/env.sh" \
        "/path/to/modelzoo_yolo/test.py --data_dir=/path/to/dataset --is_distributed=0  --pretrained=$pretraind_root$ckpt --testing_shape=416 --nms_thresh=0.65"  &

    
    echo "===="
    echo $dv
    echo $ckpt

done




