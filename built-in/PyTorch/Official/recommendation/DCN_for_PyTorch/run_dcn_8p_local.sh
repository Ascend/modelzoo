source ./test/env.sh

if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do 
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end python3.7 -u run_classification_criteo_dcn.py \
	--npu_id $i \
	--device_num 8 \
	--trainval_path='path/to/criteo_trainval.txt' \
	--test_path='path/to/criteo_test.txt' \
	--dist \
	--lr=0.0006 \
	--use_fp16 &
	done
else
   for i in $(seq 0 7)
   do
   python3.7 -u run_classification_criteo_dcn.py \
   --npu_id $i \
   --device_num 8 \
   --trainval_path='path/to/criteo_trainval.txt' \
   --test_path='path/to/criteo_test.txt' \
   --dist \
   --lr=0.0006 \
   --use_fp16 &
   done
fi