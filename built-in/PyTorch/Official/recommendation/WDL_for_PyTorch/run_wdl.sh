source ./test/env.sh
cur_path=`pwd`
export PYTHONPATH=$cur_path/../WDL_for_PyTorch:$PYTHONPATH

python3 run_classification_criteo_wdl.py  \
    --amp \
    --data_path=/data/criteo \
    --lr 0.0001 > ./train_1p.log 2>&1 &

