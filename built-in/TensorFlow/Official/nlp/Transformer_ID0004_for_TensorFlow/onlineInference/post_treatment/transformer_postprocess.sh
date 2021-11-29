#!/bin/bash
#set -x
cur_dir=`pwd`

collect_result()
{
    echo "python3 transformer_calculation_bleu_score.py --source_dir $source_dir --realFile $realFile --vocab $vocab"
    python3 transformer_calculation_bleu_score.py --source_dir $source_dir --realFile $realFile  --vocab $vocab | tee $cur_dir/../output/predict_accuracy.txt

}

source_dir="../result_Files"
realFile="../datasets/newstest2014.tok.de"
vocab="../datasets/vocab.share"

for para in $*
do
    if [[ $para == --source_dir* ]];then
        source_dir=`echo ${para#*=}`
    elif [[ $para == --realFile* ]];then
        realFile=`echo ${para#*=}`
    elif [[ $para == --vocab* ]];then
        vocab=`echo ${para#*=}`
    fi
done

echo "=======================collect_result===================="
collect_result

