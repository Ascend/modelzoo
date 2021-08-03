export TF_CPP_MIN_LOG_LEVEL=2
export RANK_SIZE=1

python3.7 npu_train.py \
--input_size=512 \
--batch_size_per_gpu=20 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=30 \
--max_steps=100000 \
--use_processed_data=True \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt