source ./test/env.sh
cur_path=`pwd`
export PYTHONPATH=$cur_path/../WDL_for_PyTorch:$PYTHONPATH

if [ $(uname -m) = "aarch64" ]
then
	for i in $(seq 0 7)
	do 
	let p_start=0+24*i
	let p_end=23+24*i
	taskset -c $p_start-$p_end $CMD python3.7 -u run_classification_criteo_wdl.py \
        --device_id $i \
        --data_path ../data/criteo/origin_data \
        --lr=0.0001 \
	      --sparse_embed_dim 4 \
	      --batch_size 1024 \
	      --epochs 3 \
        --amp \
        --device_num 8 \
        --dist > train_$i.log 2>&1 &
	done
else
    for i in $(seq 0 7)
    do
    python3.7 -u run_classification_criteo_wdl.py \
        --device_id $i \
        --data_path ../data/criteo/origin_data \
        --lr=0.0001 \
	      --sparse_embed_dim 4 \
	      --batch_size 1024 \
	      --epochs 3 \
        --amp \
        --device_num 8 \
        --dist > train_$i.log 2>&1 &
    done
fi
