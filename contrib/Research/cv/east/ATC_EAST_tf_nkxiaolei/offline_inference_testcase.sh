#! /bin/bash
obs_url="obs://modelzoo-train-atc/003_Atc_Models/nkxiaolei/EAST/east_text_detection.om"
model_path="./model/east_text_detection.om"

obsutil cp $obs_url ./model -f -r

python3.7 main.py --model=$model_path 2>&1 |tee inference.log

#关键字检查
key1="npu compute cost"

if [ `grep -c "$key1" "inference.log"` -ne '0' ] ;then   #可以根据需要调整检查逻辑
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi



