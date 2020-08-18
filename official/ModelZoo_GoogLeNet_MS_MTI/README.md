# GoogLeNet for MindSpore

GoogLeNet won the 1st place in Image Large Scale Visual Recognition Challenge (ILSVRC) 2014. 
GoogLeNet along with its Inception module evolves in the next few years, 
resulting in spotted architectures such as InceptionV1, InceptionV2, 
Inception V3 and so on. 
This repository provides a script and recipe to GoogLeNet model and 
achieve state-of-the-art performance.

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)  
  * [Default configuration](#default-configuration)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Performance](#performance)
  * [Results](#results)
    * [Training accuracy](#training-accuracy)
    * [Training performance](#training-performance)
    * [One-hour performance](#one-hour-performance)


    

## Model overview

Refer to [this paper][1] for network details.

`Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.`

[1]: https://arxiv.org/abs/1409.4842

## Default Configuration

- network architecture

  Based on Inception module, add BatchNorm layer after each Conv layer

- Preprocessing

  Input size as 224 * 224, RandomCrop, RandomHorizontalFlip, RandomColorAdjust, Normalization
  
- Hyper Parameters

  Momentum(0.9), exponential `learning rate` scheduler, initial `lr=0.1`, decrease lr by 70% every 70 epochs,
  `MaxEpoch=300`, `BatchSize=256`, `WeightDecay=0.0001`

  refer to

## Setup

The following section lists the requirements to start training the googlenet model.


### Requirements

Before training, please make sure you already have

- Ascend hardware environment
- mindspore

Apply for resources by sending [table][2]  to ascend@huawei.com.

[2]: https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx

Learn more about MindSpore with

- [MindSpore tutorial][3]
- [MindSpore API reference][4]

[3]: https://www.mindspore.cn/tutorial/zh-CN/master/index.html
[4]: https://www.mindspore.cn/api/zh-CN/master/index.html


## Quick Start Guide

### 1. Clone the respository

```
git clone xxx
cd googlenet
```

### 2. Download and preprocess the dataset

1. download training and validation dataset, such as ImageNet2012, CIFAR10 and so on.
2. extract the dataset to `train/` and `val/` respectively.
   All images within one folder have the same label.



### 3. Train

Below we offer training scripts for 8 devices and single device, respectively.
Once launched, 

- checkpoint will be saved every `ckpt_interval` steps in `ckpt_path`

- loss &accuracy will be recorded every `log_interval` steps, training log is also saved in `ckpt_path`

Example for training with 8 devices

```
python /path/to/launch.py \
--nproc_per_node=8 \
--visible_devices=0,1,2,3,4,5,6,7 \
--env_sh=/path/to/env_sh.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/train.py \
--per_batch_size=256 \
--data_dir=/path/to/dataset/train \
--is_distributed=1 \
--lr_scheduler=exponential \
--lr_epochs=70,140,210,280 \
--lr_gamma=0.3 \
--per_batch_size=256 \
--lr=0.1 \
--max_epoch=300 \
--label_smooth=1 \
--num_classes=xx
```

Example for training with single device

```
python /path/to/launch.py \
--nproc_per_node=1 \
--visible_devices=0 \
--env_sh=/path/to/env_sh.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/train.py \
--per_batch_size=256 \
--data_dir=/path/to/dataset/train \
--is_distributed=0 \
--lr_scheduler=exponential \
--lr_epochs=70,140,210,280 \
--lr_gamma=0.3 \
--per_batch_size=256 \
--lr=0.1 \
--max_epoch=300 \
--label_smooth=1 \
--num_classes=xx
```


### 4. Test

Example for inference with 8 devices, refer to `log_path` for performance results.

```
python /path/to/launch.py \
--nproc_per_node=8 \
--visible_devices=0,1,2,3,4,5,6,7 \
--env_sh=/path/to/env_sh.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/test.py \
--per_batch_size=128 \
--data_dir=/path/to/dataset/val \
--is_distributed=1 \
--pretrained=/path/to/ckpt \
--num_classes=xx \
--backbone=googlenet
--log_path=/path/to/log_dir
``` 

Example for inference with single device

```
python /path/to/launch.py \
--nproc_per_node=1 \
--visible_devices=0 \
--env_sh=/path/to/env_sh.sh \
--server_id=xx.xxx.xxx.xxx \
/path/to/test.py \
--per_batch_size=128 \
--data_dir=/path/to/dataset/val \
--is_distributed=0 \
--pretrained=/path/to/ckpt \
--num_classes=xx \
--backbone=googlenet
--log_path=/path/to/log_dir
``` 

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy

| **epochs** |   Top1_acc   |    Top5_acc |
| :--------: | :-----------: | :------ |
|     300     | 71.86%       | 90.70%  |

#### Training performance

| **NPUs** | train performance |
| :------: | :---------------: |
|    1     |   1600 img/s   |
|    8     |   13000 img/s   |

#### One-hour performance

After training one hour (refer to training parameters in ./scripts/train_8p.sh), confirm


| **items** | train performance |
| :------: | :---------------: |
|  epoch   |      >= 36         |
|  loss    |      <= 2.0        |
|  top1_acc|      >= 58.0%      |
|  top5_acc|      >= 82.0%      |








