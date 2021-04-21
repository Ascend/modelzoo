#!/bin/bash
#set -x
cur_dir=`pwd`

source_file="../datasets/newstest2014.tok.bpe.32000.en"
real_file="../datasets/newstest2014.tok.bpe.32000.de"
vocab_file="../datasets/vocab.share"

for para in $*
do
    if [[ $para == --source_file* ]];then
        source_file=`echo ${para#*=}`
    elif [[ $para == --real_file* ]];then
        real_file=`echo ${para#*=}`
    elif [[ $para == --vocab_file* ]];then
        vocab_file=`echo ${para#*=}`
    fi
done
echo "source_file:$source_file"
echo "real_file: $real_file"
echo "vocab_file: $vocab_file"
echo "===================start construct transformer NN input datas===================="
paste $source_file $real_file > test.all
python3 transformer_create_infer_data.py --input_file test.all --vocab_file $vocab_file --output_file ./newstest2014-l128-mindrecord 
