#!/bin/bash

if [ ! $1 ]; then
    $1='ppi'
fi
if [ ! $2 ]; then
    $2='meanpool'
fi

bash scripts/run_npu_unsupervised_wrapper.sh 0 1 $1 $2
