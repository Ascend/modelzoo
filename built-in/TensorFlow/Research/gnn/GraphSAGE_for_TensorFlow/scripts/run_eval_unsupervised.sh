#!/bin/bash

if [ ! $1 ]; then
    $1='ppi'
fi
if [ ! $2 ]; then
    $2='meanpool'
fi

if [ $1 -eq 'ppi' ] || [ $1 -eq 'toy-ppi' ]; then
    python3 eval_scripts/ppi_eval data/$1/$1 outputs/$1/unsup-1p/$2 test
fi
