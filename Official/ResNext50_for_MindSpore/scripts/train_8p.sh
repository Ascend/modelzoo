python /path/to/modelzoo_resnext50/launch.py --nproc_per_node=8 --visible_device=0,1,2,3,4,5,6,7 \
--env_sh=/path/to/modelzoo_resnext50/scripts/env.sh --server_id=xx.xxx.xxx.xxx /path/to/modelzoo_resnext50/train.py \
--per_batch_size=128 --data_dir=/path/to/dataset/train/ --is_distribute=1 --lr=0.4 --lr_scheduler=cosine_annealing \
--label_smooth=1 --T_max=150 --max_epoch=150 --backbone=resnext50 --warmup_epochs=1 --ckpt_interval=1250