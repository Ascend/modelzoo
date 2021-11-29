

#source env.sh
#export DUMP_GE_GRAPH=1
DATA_DIR='./data'
SAVE_DIR='./model'

mkdir -p ${SAVE_DIR}

export RANK_SIZE=1
export DEVICE_ID=0
python3 tools/train_npu.py --dataset_dir=${DATA_DIR} \
                           --char_dict_path=data/char_dict/char_dict.json \
                           --ord_map_dict_path=data/char_dict/ord_map.json \
                           --save_dir=${SAVE_DIR} \
                           --momentum=0.95 \
                           --lr=0.02 \
                           --use_nesterov=True \
                           --num_iters=3000 > training_100.log

