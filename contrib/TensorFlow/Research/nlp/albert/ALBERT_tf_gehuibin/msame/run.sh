export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}


mode=base
input_dir=../squad_v2
input_id_path=./input_ids
input_mask_path=./input_masks
segment_id_path=./segment_ids
p_mask_path=./p_masks
idx_file=./idx.txt
config=../albert_${mode}_v2/albert_config.json
output_dir=./pb_albert_${mode}_model
ckpt_dir=../output_${mode}_v2

python3.7 -m freeze_graph \
	--albert_config=${config} \
	--output_dir=${output_dir} \
	--ckpt_dir=${ckpt_dir}

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=${output_dir}/albert.pb --framework=3 --output=${output_dir}/albert --soc_version=Ascend910 \
	--input_shape="input_ids:1,384;input_mask:1,384;segment_ids:1,384;p_mask:1,384" \
	--log=info \
	--out_nodes="end_logits/TopKV2:0;end_logits/TopKV2:1;end_logits/Reshape_1:0;end_logits/Reshape_2:0;answer_class/Squeeze:0"

./msame --model ${output_dir}/albert.om --input ${input_id_path},${input_mask_path},${segment_id_path},${p_mask_path} --output ${output_dir}/output --outfmt TXT

python3.7 evaluate.py --input_dir=$input_dir \
	--idx_file=$idx_file \
	--output_dir=$output_dir

