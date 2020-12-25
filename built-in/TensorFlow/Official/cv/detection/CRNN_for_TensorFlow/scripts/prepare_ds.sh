



CWD=$(cd "$(dirname "$0")"; pwd)
echo ${CWD}
cd ${CWD}
cd ..
CWD=$(pwd)





echo ${CWD}
mkdir -p ${CWD}/data/test
mkdir -p ${CWD}/data/tfrecords

# generate tfrecords

python3 ${CWD}/tools/write_tfrecords.py --dataset_dir=${CWD}/data/mnt/ramdisk/max/90kDICT32px/ \
	--save_dir=${CWD}/data/tfrecords \
	--char_dict_path=${CWD}/data/char_dict/char_dict.json \
	--ord_map_dict_path=${CWD}/data/char_dict/ord_map.json


#tar -xvf ${CWD}/data/*IC*.tar* -C ${CWD}/data/test
#tar -xvf ${CWD}/data/*III*.tar* -C ${CWD}/data/test
#unzip ${CWD}/data/*.zip -d ${CWD}/data/test
#
#rm -rf ${CWD}/data/test/__*
#
#
#
#ROOT_DIR=${CWD}/data/test
#DATASET=svt1
#
#python3 ${CWD}/data_provider/convert_svt.py --dataset_dir=${ROOT_DIR}/${DATASET}/ \
#		       --xml_file=${ROOT_DIR}/${DATASET}/test.xml \
#		       --output_dir=${ROOT_DIR}/${DATASET}/processed \
#		       --output_annotation=${ROOT_DIR}/${DATASET}/annotation.txt \
#		       --output_lexicon=lexicon.txt 
#
#DATASET=IIIT5K
#
#python3 ${CWD}/data_provider/convert_iiit5k.py --mat_file=${ROOT_DIR}/${DATASET}/testdata.mat \
#		       --output_annotation=${ROOT_DIR}/${DATASET}/annotation.txt 
#                       
#
#
#DATASET=2003/SceneTrialTest
#
#python3 ${CWD}/data_provider/convert_ic03.py --dataset_dir=${ROOT_DIR}/${DATASET}/ \
#		       --xml_file=${ROOT_DIR}/${DATASET}/words.xml \
#		       --output_dir=${ROOT_DIR}/${DATASET}/processed \
#                       --output_annotation=${ROOT_DIR}/${DATASET}/annotation.txt
#
#python3 ${CWD}/data_provider/preprocess_ic03.py --src_ann=${ROOT_DIR}/${DATASET}/annotation.txt \
#		       --dst_ann=${ROOT_DIR}/${DATASET}/processed_annotation.txt
