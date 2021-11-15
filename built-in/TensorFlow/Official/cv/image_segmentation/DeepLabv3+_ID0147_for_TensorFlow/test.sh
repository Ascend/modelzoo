pip install tf-slim
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python deeplab/vis.py \
	--logtostderr    \ 
	--vis_split="test"    \
	--model_variant="xception_65"    \
	--atrous_rates=12    \
	--atrous_rates=24   \
	--atrous_rates=36   \
	--output_stride=8   \
	--decoder_output_stride=4   \
	--vis_crop_size="513,513"    \
	--dataset="pascal_voc_seg"   \
	--checkpoint_dir=log    \
	--vis_logdir=log_e    \
	--dataset_dir=deeplab/datasets/pascal_voc_seg/tfrecord \
	--also_save_raw_predictions=True
cd log_e
tar -zcvf results.tgz results