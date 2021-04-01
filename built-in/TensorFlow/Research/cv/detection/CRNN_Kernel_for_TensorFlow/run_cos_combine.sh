


source env.sh

export DEVICE_ID=6
python3 tools/train_npu_v2.py --dataset_dir=data/v2/ \
                           --char_dict_path=data/char_dict_bak/char_dict.json \
                           --ord_map_dict_path=data/char_dict_bak/ord_map.json \
                           --save_dir=./model/cos_combine_1 \
                           --momentum=0.95 \
                           --lr=0.015 \
			   --use_nesterov=True \
                           --num_iters=200000 >./model/combine_exp1.log &


export DEVICE_ID=1
python3 tools/train_npu_v2.py --dataset_dir=data/v2/ \
                           --char_dict_path=data/char_dict_bak/char_dict.json \
                           --ord_map_dict_path=data/char_dict_bak/ord_map.json \
                           --save_dir=./model/cos_combine_2 \
                           --momentum=0.95 \
                           --lr=0.015 \
                           --use_nesterov=True \
                           --num_iters=300000 >./model/combine_exp2.log &


export DEVICE_ID=2
python3 tools/train_npu_v2.py --dataset_dir=data/v2/ \
                           --char_dict_path=data/char_dict_bak/char_dict.json \
                           --ord_map_dict_path=data/char_dict_bak/ord_map.json \
                           --save_dir=./model/cos_combine_3 \
                           --momentum=0.95 \
                           --lr=0.02 \
                           --use_nesterov=True \
                           --num_iters=400000 >./model/combine_exp3.log &

export DEVICE_ID=3
python3 tools/train_npu_v2.py --dataset_dir=data/v2/ \
                           --char_dict_path=data/char_dict_bak/char_dict.json \
                           --ord_map_dict_path=data/char_dict_bak/ord_map.json \
                           --save_dir=./model/cos_combine_4 \
                           --momentum=0.97 \
                           --lr=0.015 \
                           --use_nesterov=True \
                           --num_iters=200000 >./model/combine_exp4.log &

export DEVICE_ID=4
python3 tools/train_npu_v2.py --dataset_dir=data/v2/ \
                           --char_dict_path=data/char_dict_bak/char_dict.json \
                           --ord_map_dict_path=data/char_dict_bak/ord_map.json \
                           --save_dir=./model/cos_combine_5 \
                           --momentum=0.95 \
                           --lr=0.02 \
                           --use_nesterov=True \
                           --num_iters=600000 >./model/combine_exp5.log &
export DEVICE_ID=5
python3 tools/train_npu_v2.py --dataset_dir=data/v2/ \
                           --char_dict_path=data/char_dict_bak/char_dict.json \
                           --ord_map_dict_path=data/char_dict_bak/ord_map.json \
                           --save_dir=./model/cos_combine_6 \
                           --momentum=0.95 \
                           --lr=0.017 \
                           --use_nesterov=True \
                           --num_iters=400000 >./model/combine_exp6.log &

