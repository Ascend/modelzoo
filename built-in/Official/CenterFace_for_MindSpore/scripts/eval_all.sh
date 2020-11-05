#!/bin/sh
root=$PWD
save_path=$root/output/centerface/
ground_truth_path=$root/dataset/centerface/ground_truth
#for i in $(seq start_epoch end_epoch+1)
for i in $(seq 89 200)
do
    python ../eval.py --pred=$save_path$i --gt=$ground_truth_path &
    sleep 10
done
wait
