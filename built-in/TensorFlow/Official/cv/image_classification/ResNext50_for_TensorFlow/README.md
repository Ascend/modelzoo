# ResNext50_for_TensorFlow

## 目录

* [概述](#概述)
* [要求](#要求)
* [默认配置](#默认配置)
* [快速上手](#快速上手)
  * [准备数据集](#准备数据集)
  * [关键配置修改](#关键配置修改)
  * [运行示例](#运行示例)
    * [训练](#训练)
    * [推理](#推理)
* [高级](#高级)
  * [脚本参数](#脚本参数) 



## 概述

ResNeXt网络在ResNet基础上进行了优化，同时采用Vgg/ResNet堆叠的思想和Inception的split-transform-merge思想，把单路卷积转变成了多个支路的多个卷积。ResNeXt结构可以在不增加参数复杂度的前提下提高准确率，同时还减少了超参数的数量。ResNeXt有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。Ascend本次提供的是50-layer的ResNeXt-50网络。

参考论文：Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.Aggregated Residual Transformations for Deep Neural Networks

## 要求

- 安装有昇腾AI处理器的硬件环境

- 下载并预处理ImageNet2012，CIFAR10或Flower数据集以进行培训和评估。


## 默认配置

- 训练数据集预处理（以ImageNet-Train训练集为例，仅作为用户参考示例）：

  图像的输入尺寸为224*224

  图像输入格式：TFRecord

  数据集大小：1281167

- 测试数据集预处理（以ImageNet-Val验证集为例，仅作为用户参考示例）：

  图像的输入尺寸为224*224

  图像输入格式：TFRecord

  验证集大小：50000

- 训练超参（8卡）：

  Batch size: 32

  Momentum: 0.9

  loss_scale：1024

  LR scheduler: cosine

  Learning rate(LR): 0.1

  learning_rate_end: 0.000001

  warmup_epochs: 5

  train epoch: 120

## 快速上手

### 准备数据集

- 请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括ImageNet2012，CIFAR10、Flower等。
- 以ImageNet2012举例，训练集和验证集图片统一放到“data/resnext50/imagenet_TF”目录下。

### 关键配置修改

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)



### 运行示例

#### 训练

- 容器场景

修改`testscript`目录下`Resnext50_*_docker.sh`文件，将对应容器镜像名修改为实际名称。

`Resnext50_1p_docker.sh`脚本文件:

```
# user testcase
casecsv="case_resnext50.csv"
casenum=1

# docker or host
exectype="docker"

ostype=`uname -m`
if [ x"${ostype}" = xaarch64 ];
then
    # arm
    dockerImage="ubuntu_arm:18.04"
else
    # x86
    dockerImage="ubuntu:16.04"
fi
```
   
`Resnext50_8p_docker`脚本文件:

```                   
# user testcase
casecsv="case_resnext50.csv"
casenum=8

# docker or host
exectype="docker"

ostype=`uname -m`
if [ x"${ostype}" = xaarch64 ];
then
    # arm
    dockerImage="ubuntu_arm:18.04"
else
    # x86
    dockerImage="ubuntu:16.04"
fi
```

执行训练脚本:

1P训练指令（脚本位于`testscript/Resnext50_1p_docker.sh`）

```
./Resnext50_1p_docker.sh
```

8P训练指令（脚本位于`testscript/Resnext50_8p_docker.sh`）

```
./Resnext50_8p_docker.sh
```


- 物理机场景

修改`case_resnext50_host.csv`中，路径为脚本所在的绝对路径地址

```
/home/models/training_shop/03-code/ModelZoo_ResNext50_TF_MTI/code/resnext50_train/mains/res50.py
--model_dir=/home/models/training_shop/03-code/ModelZoo_ResNext50_TF_MTI/d_solution/ckpt${DEVICE_ID}
```

修改`ModelZoo_ResNext50_TF_MTI\code\resnext50_train\configs` 中`res50_32bs_1p_host` 和 `res50_32bs_8p_host`文件:

配置数据集的绝对路径地址

```
'data_url':  'file:///home/models/training_shop/03-code/ModelZoo_ResNext50_TF_MTI/data/resnext50/imagenet_TF',
```

配置checkpoint文件的路径

```
'ckpt_dir': '/home/models/training_shop/03-code/ModelZoo_ResNext50_TF_MTI/d_solution/ckpt0',
```

执行训练脚本:

1P训练指令（脚本位于`testscript/Resnext50_1p_host.sh`）

```
./Resnext50_1p_host.sh
```

8P训练指令（脚本位于`testscript/Resnext50_8p_host.sh`）

```
./Resnext50_8p_host.sh
```


#### 推理

在120 epoch训练执行完成后，脚本会自动执行验证流程


## 高级

### 脚本参数

```
--rank_size              使用NPU卡数量，默认：单P 配置1，8P 配置8
--mode                   运行模式，默认train；可选：train，evaluate
--max_train_steps        训练次数，单P 默认：10000
--iterations_per_loop    NPU运行时，device端下沉次数，默认：1000
--eval                   训练结束后，是否启动验证流程。默认：单P False，8P True
--num_epochs             训练epoch次数， 默认：单P None，8P 120 
--data_url               数据集路径，默认：data/resnext50/imagenet_TF
--ckpt_dir               验证时checkpoint文件地址 默认：/d_solution/ckpt0
--lr_decay_mode          学习率方式，默认：cosine  
--learning_rate_maximum  初始学习率，默认：0.1
--learning_rate_end      结束学习率：默认：0.000001
--batch_size             每个NPU的batch size，默认：32
--warmup_epochs          初始warmup训练epoch数，默认：5
--momentum               动量，默认：0.9
```