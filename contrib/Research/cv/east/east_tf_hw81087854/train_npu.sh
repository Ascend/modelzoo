python npu_train.py \
--input_size=512 \
--batch_size_per_gpu=14 \
--checkpoint_path=./checkpoint/ \
--text_scale=512 \
--training_data_path=./ocr/icdar2015/ \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--pretrained_model_path=./pretrain_model/resnet_v1_50.ckpt
