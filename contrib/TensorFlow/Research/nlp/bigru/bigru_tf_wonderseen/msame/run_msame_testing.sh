rm -rf golden 
mkdir golden
cd golden
mkdir references
cd ..

rm -rf output_npu 
rm -rf output_offline
rm -rf input

mkdir output_npu
mkdir output_offline
mkdir input


. transform_ckpt_to_om.sh
. process_phrase_1.sh
. process_phrase_2.sh
