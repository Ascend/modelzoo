#!/bin/bash

for ((i=0; i<10; i++));
do
rm ge_onnx_0$i*;
rm ge_proto_0$i*;
rm aicpu_proto_$i*;
rm aicpu_proto_0$i*;
done


for ((i=10; i<100; i++));
do
rm ge_onnx_$i*;
rm ge_proto_$i*;
rm aicpu_proto_$i*;
done

rm After*
rm Before*
rm TF_*
