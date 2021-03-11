export ckpt_file=$1
export output_path=./output

mkdir ${output_path}
rm -rf ${output_path}/*

python3 main/demo.py --checkpoint_path=${ckpt_file} --test_data_path=./data/dataset/IC13_test/ --output_path=${output_path}

cd ${output_path}
zip results.zip res_img_*.txt
cd ..
python3 test/script.py -g=./test/gt.zip -s=${output_path}/results.zip