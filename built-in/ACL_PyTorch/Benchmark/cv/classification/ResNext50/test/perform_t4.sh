#!/bin/bash

# T4上执行：
./trtexec --onnx=resnext50.onnx --fp16 --shapes=image:1x3x224x224 --threads
./trtexec --onnx=resnext50.onnx --fp16 --shapes=image:16x3x224x224 --threads
