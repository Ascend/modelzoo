python /PATH/TO/MODEL_ZOO_CODE/launch.py \
--nproc_per_node=8 --visible_devices=0,1,2,3,4,5,6,7 \
--env_sh=/PATH/TO/MODEL_ZOO_CODE/scripts/env.sh \
--server_id=X.X.X.X \
/PATH/TO/MODEL_ZOO_CODE/train.py  \
--backbone googlenet --per_batch_size 256 --max_epoch 320 --lr 0.1 --lr_scheduler exponential \
--lr_epochs 70,140,210,280 --lr_gamma 0.3 --label_smooth 1 \
--ckpt_path /PATH/TO/OUTPUT --data_dir /PATH/TO/imagenet/train --is_distributed 1