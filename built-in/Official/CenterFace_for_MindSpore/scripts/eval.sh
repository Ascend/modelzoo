#!/bin/bash
root=$PWD
save_path=$root/output/centerface/
ground_truth_path=$root/dataset/centerface/ground_truth
echo "start eval"
python ../eval.py --pred=$save_path --gt=$ground_truth_path
echo "end eval"
