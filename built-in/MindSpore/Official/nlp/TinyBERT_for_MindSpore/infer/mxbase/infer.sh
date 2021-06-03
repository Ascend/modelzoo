#!/bin/bash
bash build.sh

./main --model=../data/model/tinybert.om --input_file=../data/dataset/input_file.txt --vocab_txt=../data/dataset/vocab.txt \
       --eval_labels_file=../data/dataset/eval_labels.txt \
       --max_seq_length=128 --device=0
