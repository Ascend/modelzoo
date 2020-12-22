

LOG_NAME=$2
CKPT_DIR=$1
DEVICE_ID=$3

CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${CURRENT_DIR}
cd ..
CWD=$(pwd)

# checkpoint path valid
if  [ x"${CKPT_DIR}}" = x ] ;
then
    echo "No directory provided , exiting "
    exit
else  
    echo "CHECKPOINT DIRECTORY: ${CKPT_DIR}"
fi


# save result to log file 
if  [ x"${LOG_NAME}" = x ] ;
then
    LOG_FILE="test_result.txt"    
else
    LOG_FILE=${LOG_NAME}
fi

echo "LOGS ARE EXPORTED TO FILE: ${LOG_FILE}"


if  [ x"${DEVICE_ID}" = x ] ;
then
    DEVICE_ID="test_result.txt"
else
    DEVICE_ID=${DEVICE_ID}
fi


echo "=================================" >>${LOG_FILE}
echo "Test SVT datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

DATASET_ROOT=${CWD}/data/test

DATASET_DIR="${DATASET_ROOT}/svt1/processed/"
ANNOTATION="${DATASET_ROOT}/svt1/annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}


python3 ${CWD}/tools/eval_ckpt.py --weights_path=${CKPT_DIR} \
       	 --device_id=${DEVICE_ID} \
	 --scripts=${CWD}/tools/other_dataset_evaluate_shadownet.py \
	 --dataset_dir=${DATASET_DIR} \
	 --root_dir=${CWD} \
	 --annotation_file=${ANNOTATION} >> ${LOG_FILE}




echo "=================================" >>${LOG_FILE}
echo "Test IIIT5K datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

DATASET_DIR="${DATASET_ROOT}/IIIT5K/"
ANNOTATION="${DATASET_ROOT}/IIIT5K/annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}


python3 ${CWD}/tools/eval_ckpt.py --weights_path=${CKPT_DIR} \
         --device_id=${DEVICE_ID} \
         --scripts=${CWD}/tools/other_dataset_evaluate_shadownet.py \
         --dataset_dir=${DATASET_DIR} \
	 --root_dir=${CWD} \
         --annotation_file=${ANNOTATION} >> ${LOG_FILE}




echo "=================================" >>${LOG_FILE}
echo "Test IC03 datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

DATASET_DIR="${DATASET_ROOT}/2003/SceneTrialTest/processed"
ANNOTATION="${DATASET_ROOT}/2003/SceneTrialTest/processed_annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}


python3 ${CWD}/tools/eval_ckpt.py --weights_path=${CKPT_DIR} \
         --device_id=${DEVICE_ID} \
         --scripts=${CWD}/tools/other_dataset_evaluate_shadownet.py \
         --dataset_dir=${DATASET_DIR} \
	 --root_dir=${CWD} \
         --annotation_file=${ANNOTATION} >> ${LOG_FILE}



