

LOG_NAME=$2
CKPT_DIR=$1
DEVICE_ID=$3


# checkpoint path valid
if  [ x"${CKPT_DIR}}" = x ] ;
then
    echo "No directory provided , exiting "
    exit
else  
    echo "checkpoint directory: ${CKPT_DIR}"
fi


# save result to log file 
if  [ x"${LOG_NAME}" = x ] ;
then
    LOG_FILE="result.txt"    
else
    LOG_FILE=${LOG_NAME}
fi

echo "LOGS ARE EXPORTED TO FILE: ${LOG_FILE}"




echo "=================================" >>${LOG_FILE}
echo "Test SVT datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

#DATASET_DIR="/data/m00536736/modelzoo/OCR/datasets/svt_converted/processed/"
DATASET_DIR="./data/test/svt1/processed"
#ANNOTATION="/data/m00536736/modelzoo/OCR/datasets/svt_converted/annotation.txt"
ANNOTATION="./data/test/svt1/annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}


python3 eval_ckpt.py --weights_path=${CKPT_DIR} \
       	 --device_id=0 \
	 --scripts=tools/other_dataset_evaluate_shadownet.py \
	 --dataset_dir=${DATASET_DIR} \
	 --annotation_file=${ANNOTATION} >> ${LOG_FILE}


echo "=================================" >>${LOG_FILE}
echo "Test IC13 datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

#DATASET_DIR="/data/m00536736/modelzoo/OCR/datasets/2013/Challenge2_Test_Task3_Images/"
DATASET_DIR="./data/test/2003/SceneTrialTest/processed"
#ANNOTATION="/data/m00536736/modelzoo/OCR/datasets/2013/processed_annotation.txt"
ANNOTATION="./data/test/2003/SceneTrialTest/processed_annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}

python3 eval_ckpt.py --weights_path=${CKPT_DIR} \
         --device_id=0 \
         --scripts=tools/other_dataset_evaluate_shadownet.py \
         --dataset_dir=${DATASET_DIR} \
         --annotation_file=${ANNOTATION} >> ${LOG_FILE}



echo "=================================" >>${LOG_FILE}
echo "Test IIIT5K datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

#DATASET_DIR="/data/m00536736/modelzoo/OCR/datasets/IIIT5K/"
DATASET_DIR="./data/test/IIIT5K/"
#ANNOTATION="/data/m00536736/modelzoo/OCR/datasets/IIIT5K/annotation.txt"
ANNOTATION="/./data/test/IIIT5K/annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}


python3 eval_ckpt.py --weights_path=${CKPT_DIR} \
         --device_id=0 \
         --scripts=tools/other_dataset_evaluate_shadownet.py \
         --dataset_dir=${DATASET_DIR} \
         --annotation_file=${ANNOTATION} >> ${LOG_FILE}




echo "=================================" >>${LOG_FILE}
echo "Test IC03 datatset " >> ${LOG_FILE}
echo "=================================" >>${LOG_FILE}

#DATASET_DIR="/data/m00536736/modelzoo/OCR/datasets/2003/SceneTrialTest/processed"
DATASET_DIR="./data/test/2003/SceneTrialTest/processed"
#ANNOTATION="/data/m00536736/modelzoo/OCR/datasets/2003/SceneTrialTest/processed_annotation.txt"
ANNOTATION="./data/test/2003/SceneTrialTest/processed_annotation.txt"
echo "dataset: ${DATASET_DIR}" >> ${LOG_FILE}
echo " anntation: ${ANNOTATION}" >>${LOG_FILE}


python3 eval_ckpt.py --weights_path=${CKPT_DIR} \
         --device_id=0 \
         --scripts=tools/other_dataset_evaluate_shadownet.py \
         --dataset_dir=${DATASET_DIR} \
         --annotation_file=${ANNOTATION} >> ${LOG_FILE}



