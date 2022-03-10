pip install tf-slim
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python deeplab/train.py    \
	--logtostderr    \
	--training_number_of_steps=200000    \
	--train_split="trainval"    \
	--model_variant="xception_65"   \
	--atrous_rates=6    \
	--atrous_rates=12    \
	--atrous_rates=18    \
	--output_stride=16    \
	--decoder_output_stride=4     \
	--train_crop_size="513,513"    \
	--train_batch_size=16    \
	--dataset="pascal_voc_seg"     \
	--train_logdir=log    \
	--dataset_dir=deeplab/datasets/pascal_voc_seg/tfrecord    \
	--tf_initial_checkpoint=cp/xception_65_coco_pretrained/x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt \
	--fine_tune_batch_norm=False