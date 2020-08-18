python /path/to/modelzoo_resnext50/launch.py --nproc_per_node=1 --visible_device=1 \
--env_sh=/path/to/modelzoo_resnext50/scripts/env.sh --server_id=xx.xxx.xxx.xxx \
/path/to/modelzoo_resnext50/train.py --per_batch_size=128 --data_dir=/path/to/dataset/train/ --is_distribute=0 \
--lr_scheduler=cosine_annealing --lr=0.05 --label_smooth=1 --T_max=150 --max_epoch=150 --backbone=resnext50 --warmup_epochs=1