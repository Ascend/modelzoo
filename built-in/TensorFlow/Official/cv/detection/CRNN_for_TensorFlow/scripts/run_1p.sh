EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${EXEC_DIR}
cd ..
EXEC_DIR=$(pwd)
echo ${EXEC_DIR}

source ${EXEC_DIR}/scripts/env.sh


DATA_DIR='data/'
SAVE_DIR='results/1p'
export RANK_SIZE=1
export DEVICE_ID=0
mkdir -p ${SAVE_DIR}/${DEVICE_ID}

python3 ${EXEC_DIR}/tools/train_npu.py --dataset_dir=${EXEC_DIR}/${DATA_DIR} \
                           --char_dict_path=${EXEC_DIR}/data/char_dict/char_dict.json \
                           --ord_map_dict_path=${EXEC_DIR}/data/char_dict/ord_map.json \
                           --save_dir=${SAVE_DIR}/${DEVICE_ID} \
                           --momentum=0.95 \
                           --lr=0.02 \
                           --use_nesterov=True \
                           --num_iters=600000 >${SAVE_DIR}/training.log 2>&1 &


