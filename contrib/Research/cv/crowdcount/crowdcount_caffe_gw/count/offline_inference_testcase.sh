#! /bin/bash
#start inference
./out/main ./data/ 2>&1 > inference.log

expect_time=240
#性能检查耗时
avg_time=`grep "Inference average time without first time:" inference.log | awk '{print $8}'`
echo "Average inference time is $avg_time ms, expect time is <$expect_time ms"
if [ $avg_time -lt $expect_time ];then
   echo "Run testcase success!"
else
   echo "Run testcase failed!"
fi

