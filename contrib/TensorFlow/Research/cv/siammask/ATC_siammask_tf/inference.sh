model="./model/siammask.om"
dataset="VOT2016"

echo $model
echo $dataset

python3.7  ./tool/test_npu_inference.py \
    --config ./config/config.json \
    --resume $model \
    --mask \
    --dataset $dataset

python3.7 tool/eval.py --dataset $dataset --tracker_prefix C --result_dir ./test/$dataset
#python3.7 tool/eval.py --dataset VOT2016 --tracker_prefix C --result_dir ./test/VOT2016