source ./test/env.sh
cur_path=`pwd`
export PYTHONPATH=$cur_path/../WDL_for_PyTorch:$PYTHONPATH

python3 run_classification_criteo_wdl.py  \
    --amp \
    --data_path=../data/criteo/origin_data \
    --batch_size 4096 \
    --lr 0.0009 > ./train_1p.log 2>&1 &