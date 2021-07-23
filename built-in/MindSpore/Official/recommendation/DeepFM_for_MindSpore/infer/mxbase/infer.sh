#!/bin/bash

bash build.sh

./main --feat_ids ../data/feat_ids.bin --feat_vals ../data/feat_vals.bin --sample_num 4518000 > ../data/preds.txt