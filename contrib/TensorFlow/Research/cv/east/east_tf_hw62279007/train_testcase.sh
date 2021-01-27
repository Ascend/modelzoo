python multigpu_train.py \
		--checkpoint_path='./checkpoint' \
		--text_scale=512 \
		--training_data_path=$icdar2015_train \
		--geometry=RBOX \
		--learning_rate=0.0001 \
		--num_readers=24 \
		--max_steps=20 \
		--save_checkpoint_steps=10 >train.log 2>&1
