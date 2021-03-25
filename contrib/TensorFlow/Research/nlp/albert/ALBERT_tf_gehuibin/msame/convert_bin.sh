input_path=../squad_v2 #原始输入文件夹

#多输入的.bin格式数据
input_id_path=./input_ids
input_mask_path=./input_masks
segment_id_path=./segment_ids
p_mask_path=./p_masks

if [ ! -d ${input_id_path} ]; then
  mkdir ${input_id_path}
fi
if [ ! -d ${input_mask_path} ]; then
  mkdir ${input_mask_path}
fi
if [ ! -d ${segment_id_path} ]; then
  mkdir ${segment_id_path}
fi
if [ ! -d ${p_mask_path} ]; then
  mkdir ${p_mask_path}
fi
python3.7 convert_bin.py --input_dir ${input_path}

