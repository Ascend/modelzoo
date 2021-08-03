export output_dir=./output
export ckpt_dir=./checkpoint/
export test_data=./ocr/ch4_test_images

mkdir  ${output_dir}
rm -rf ${output_dir}/*

python3.7 eval.py \
--test_data_path=${test_data} \
--checkpoint_path=${ckpt_dir} \
--output_dir=${output_dir}

cd ${output_dir}
zip results.zip res_img_*.txt
cd ../

python3.7 evaluation/script.py -g=./evaluation/gt.zip -s=${output_dir}/results.zip