#!/bin/bash
cur_dir=`pwd`

cd ${cur_dir}/D0
bash ./scripts/run_1p_train.sh 0 8 &

cd ${cur_dir}/D1
bash ./scripts/run_1p_train.sh 1 8 &

cd ${cur_dir}/D2
bash ./scripts/run_1p_train.sh 2 8 &

cd ${cur_dir}/D3
bash ./scripts/run_1p_train.sh 3 8 &

cd ${cur_dir}/D4
bash ./scripts/run_1p_train.sh 4 8 &

cd ${cur_dir}/D5
bash ./scripts/run_1p_train.sh 5 8 &

cd ${cur_dir}/D6
bash ./scripts/run_1p_train.sh 6 8 &

cd ${cur_dir}/D7
bash ./scripts/run_1p_train.sh 7 8
