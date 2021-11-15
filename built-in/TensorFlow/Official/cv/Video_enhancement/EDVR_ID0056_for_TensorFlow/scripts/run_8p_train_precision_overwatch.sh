#!/bin/bash
cur_dir=`pwd`

cd ${cur_dir}/D0
bash ./scripts/run_1p_train_precision_overwatch.sh 0 8 &

cd ${cur_dir}/D1
bash ./scripts/run_1p_train_precision_overwatch.sh 1 8 &

cd ${cur_dir}/D2
bash ./scripts/run_1p_train_precision_overwatch.sh 2 8 &

cd ${cur_dir}/D3
bash ./scripts/run_1p_train_precision_overwatch.sh 3 8 &

cd ${cur_dir}/D4
bash ./scripts/run_1p_train_precision_overwatch.sh 4 8 &

cd ${cur_dir}/D5
bash ./scripts/run_1p_train_precision_overwatch.sh 5 8 &

cd ${cur_dir}/D6
bash ./scripts/run_1p_train_precision_overwatch.sh 6 8 &

cd ${cur_dir}/D7
bash ./scripts/run_1p_train_precision_overwatch.sh 7 8
