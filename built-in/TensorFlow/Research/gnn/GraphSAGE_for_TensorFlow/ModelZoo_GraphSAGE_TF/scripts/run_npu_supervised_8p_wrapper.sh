#!/bin/bash

DATASET=$1
MODEL=$2

scripts/run_npu_supervised_wrapper.sh 0 8 $@ &
scripts/run_npu_supervised_wrapper.sh 1 8 $@ &
scripts/run_npu_supervised_wrapper.sh 2 8 $@ &
scripts/run_npu_supervised_wrapper.sh 3 8 $@ &
scripts/run_npu_supervised_wrapper.sh 4 8 $@ &
scripts/run_npu_supervised_wrapper.sh 5 8 $@ &
scripts/run_npu_supervised_wrapper.sh 6 8 $@ &
scripts/run_npu_supervised_wrapper.sh 7 8 $@ &
