echo "start train model"
CUDA_VISIBLE_DEVICES=0 python -u ./train/main.py --config_name "mmoe_config" --tag "mmoe_transformer" --train_with_evaluate
echo "finish train model"

