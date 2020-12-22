EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${EXEC_DIR}


cd ..
EXEC_DIR=$(pwd)
echo ${EXEC_DIR}
RESULTS=results/8p

DEVICE_ID=$1
if  [ x"${DEVICE_ID}" = x ] ;
then
    echo "turing train fail" >> ${EXEC_DIR}/results/train_${DEVICE_ID}.log
    exit
else
    export DEVICE_ID=${DEVICE_ID}
fi

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8 ))
export DEVICE_INDEX=${DEVICE_INDEX}

echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

#mkdir exec path
mkdir -p ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}
cd ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}

env > ${EXEC_DIR}/results/env_${DEVICE_ID}.log


DATA_DIR='data/'
SAVE_DIR='./'

ITERATIONS=240000
LOG_FILE=training.log

python3 ${EXEC_DIR}/tools/train_npu.py --dataset_dir=${EXEC_DIR}/${DATA_DIR} \
                           --char_dict_path=${EXEC_DIR}/data/char_dict/char_dict.json \
                           --ord_map_dict_path=${EXEC_DIR}/data/char_dict/ord_map.json \
                           --save_dir=${SAVE_DIR}/ \
                           --momentum=0.95 \
                           --lr=0.08 \
                           --use_nesterov=True \
			   --warmup_step=8000 \
                           --num_iters=${ITERATIONS} >> ${LOG_FILE} 2>&1 
   


if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${DEVICE_ID}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${DEVICE_ID}.log
fi

