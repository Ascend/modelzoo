#!/bin/bash

if [ ! $1 ]; then
    $1='ppi'
fi
if [ ! $2 ]; then
    $2='meanpool'
fi

bash scripts/run_npu_supervised_8p_wrapper.sh $1 $2
