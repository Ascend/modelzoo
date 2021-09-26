# ResNet in TensorFlow On NPU
---

# Classification Model
## Overview
1. This is an implementation of the ResNet101 model as described in the [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) paper. 
2. Current implementation is based on the code from the TensorFlow Official implementation in the [tensorflow/models Repo](https://github.com/tensorflow/models).

## Introduction
1. ResNet is a relatively good network for classification problems in the ImageNet competition. It introduces the concept of residual learning. It protects the integrity of information by adding direct connection channels, solves problems such as information loss, gradient disappearance, and gradient explosion. The network can also be trained. ResNet has different network layers, commonly used are 18-layer, 34-layer, 50-layer, 101-layer, 152-layer. 
2. Ascend provides the V1.5 version of the 50-layer ResNet network this time. The difference between the V1.5 version of the ResNet network and the V1 version is that in the bottleneck module, the V1 version is set stride=2 in the first 1x1 convolutional layer, and V1.5 sets stride=2 in the 3x3 convolutional layer.

## Dataset
We have used the ImageNet dataset as an example here, you can use mnist or your own dataset to modify and adapt.
We use [build_imagenet_data](https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/slim/datasets/build_imagenet_data.py) to build record for training.

## Running Code
### Config the env paramater
check if path '/usr/local/HiAI' or ''/usr/local/Ascend' is existed or not.
modify '/usr/local/HiAI' to the actual path in scripts/run.sh

### Train and evaluate model
[imagenet_main.py](r1/resnet/imagenet_main.py) is the Entry Python script.
[resnet_run_loop.py](r1/resnet/resnet_run_loop.py) is the Main Python script.

### Check your rank_table
default rank_table setting in [configs](r1/resnet/configs) is usrd for X86.
if you use aach64, please modify board_id from "0x0000" -->

To train and evaluate the model, issue the following command:
```
# for single training
bash ./scripts/train_1p.sh
# for multi training 
bash ./scripts/train_8p.sh
```

Default Args:
- Batch size: 128
- Momentum: 0.9
- LR scheduler: cosine
- Learning rate(LR): 0.064
- loss scale: 512
- Weight decay: 0.0001
- Label smoothing: 0.1
- train epoch: 90

There are other arguments about models and training process. Use the `--help` or `-h` flag to get a full list of possible arguments with detailed descriptions.

### Train and evaluate result
- 1 NPU
    - Train performance：109ms/step，1170images/sec.
- 8 NPU
    - Train performance：109ms/step，9390images/sec.
- best result
    - Accuracy(Top1): 79.03 
    - Accuracy(Top5): 94.53

### More 

#### modify file
- The npu modify file list as follows:
- DaVinci npu platform adaptation code,including
   1.r1/resnet/imagenet_main.py 
   2.r1/resnet/resnet_model.py
   3.r1/resnet/resnet_run_loop.py
   4.utils/flags/_base.py

#### FileTree Intro
- Main Dir
    - ./r1/resnet
- Single NPU Training Shell
    - npu_train_1p_test.sh
- Multi NPU(8p) Training Shell
    - npu_train_8p_test.sh
- Log Info
    - STDOUT nohup.out
    - Performance perf.log
    