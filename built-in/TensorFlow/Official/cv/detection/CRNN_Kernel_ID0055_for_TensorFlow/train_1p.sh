EXEC_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${EXEC_DIR}

RESULTS=results/1p


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

#export RANK_ID=${DEVICE_ID}
echo $DEVICE_INDEX
echo $RANK_ID
echo $DEVICE_ID

#mkdir exec path
mkdir -p ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}
cd ${EXEC_DIR}/${RESULTS}/${DEVICE_ID}

#mkdir -p ${EXEC_DIR}/results

env > ${EXEC_DIR}/results/env_${DEVICE_ID}.log


DATA_DIR='/home/h00271187/CRNN_for_TensorFlow/data'
SAVE_DIR='./model'


iters=$2
lr=$3

if  [ x"${lr}" = x ] ;
then lr=0.015
fi

if [ x"${iters}" = x ] ;
then 
	iters=1000
fi


python3 ${EXEC_DIR}/tools/train_npu.py --dataset_dir=${DATA_DIR} \
                           --char_dict_path=${EXEC_DIR}/data/char_dict/char_dict.json \
                           --ord_map_dict_path=${EXEC_DIR}/data/char_dict/ord_map.json \
                           --save_dir=${SAVE_DIR} \
                           --momentum=0.9 \
                           --lr=${lr} \
                           --use_nesterov=False \
                           --num_iters=${iters} >> training.log 


#python3 ${EXEC_DIR}/tools/train_npu.py --root_dir=${EXEC_DIR} >>loss.txt 

if [ $? -eq 0 ] ;
then
    echo "turing train success" >> ${EXEC_DIR}/${RESULTS}/train_${DEVICE_ID}.log
else
    echo "turing train fail" >> ${EXEC_DIR}/${RESULTS}/train_${DEVICE_ID}.log
fi

