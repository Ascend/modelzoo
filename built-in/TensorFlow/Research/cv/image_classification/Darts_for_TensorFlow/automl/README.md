# DARTS for TensorFlow

## 目录

* [概述](#概述)
* [要求](#要求)
* [默认配置](#默认配置)
* [快速上手](#快速上手)
  * [准备数据集](#准备数据集)
  * [关键配置修改](#关键配置修改)
  * [运行示例](#运行示例)
    * [训练](#训练)
* [高级](#高级)
  * [脚本参数](#脚本参数)  

## 概述 


DARTS(Differentiable ARchiTecture Search)是最早提出基于梯度下降算法实现神经网络架构搜索（Neural Architecture Search, NAS）的算法之一，其通过softmax函数将离散的搜索空间松弛化为连续的搜索空间，从而允许在架构搜索时使用梯度下降。在整个搜索过程中，DARTS交替优化网络权重和架构权重，并且还进一步探讨了使用二阶优化来代替一阶优化的提高性能的可能性。相比如早期基于强化学习和进化算法的NAS算法，DARTS可以在更短时间和更少计算资源的情况下找到类似甚至更好的网络架构。本次提供的是基于TensorFlow框架实现的DARTS代码。

参考论文：Hanxiao Liu, Karen Simonyan, Yiming Yang. DARTS: Differentiable Architecture Search, ICLR 2019.

参考实现：https://github.com/quark0/darts


## 要求

- 安装有昇腾AI处理器的硬件环境

- 下载并预处理ImageNet2012，CIFAR10或Flower数据集以进行培训和评估 


## 默认配置

- 网络结构

  网络整体结构如右：['PreOneStem','normal','normal','reduce','normal','normal','reduce','normal','normal','classifier']，其中'PreOneStem'是固定结构，其输入通道数为16。'normal'和'reduce'分别是需要搜索的单元结构，'classifier'是最后的全连接层。

- 训练数据集预处理（当前代码以CIFAR-10训练集为例，仅作为用户参考示例）：

  图像的输入尺寸为32*32

  图像四周分别填充4个像素

  随机裁剪图像尺寸

  随机水平翻转图像

  根据CIFAR-10数据集通用的平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理（当前代码以CIFAR-10验证集为例，仅作为用户参考示例）：

  图像的输入尺寸为32*32

  根据CIFAR-10数据集通用的平均值和标准偏差对输入图像进行归一化

- 训练超参（单卡）：

  Batch size: 96

  Momentum: 0.9

  LR scheduler: cosine

  Learning rate (LR): 0.025

  Weight decay: 0.0001

  Label smoothing: 0.1

  Train epoch: 600

## 快速上手

### 准备数据集

- 请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括ImageNet2012，CIFAR10、Flower等，包含train和val两部分。

- 将训练集和验证集图片置于 `/root/datasets/cifar10/cifar-10-batches-bin` 目录中。

- 当前提供的训练脚本中，是以CIFAR-10数据集为例，训练过程中进行数据预处理操作，请用户使用该脚本之前自行修改训练脚本中的数据集加载和预处理方法。

### 关键配置修改

 启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


- 安装Vega库

首先进入 `automl` 文件夹

```
cd /path/to/automl
```

`automl`文件夹下有一个 `deploy` 目录，该目录下的一个 `shell` 文件用于安装依赖库，运行如下命令：

```
bash deploy/install_dependencies.sh
```

将 `automl` 文件夹加入环境变量，这样才完成了vega库的安装

```
export PYTHONPATH=/path/to/automl/:$PYTHONPATH
```

- 修改配置文件参数

`automl` 下的 `examples` 目录预先设置了不同算法的配置文件，在配置文件中进行参数配置，搜索模型、训练模型可参考以下配置文件：`automl/examples/nas/darts_cnn/darts_tf.yml`

### 运行示例

#### 训练


`automl` 下的 `examples` 目录提供了运行不同算法 `shell` 文件。你首先需要进入 `/path/to/automl/examples` 文件夹，之后你可以直接通过运行如下命令运行DARTS算法

```
cd /path/to/automl/examples
bash run_darts.sh 0
```

上面bash命令后的 0 用于指定设备ID，适用于单卡运行的场景



## 高级

### 脚本参数

默认的数据集路径是 `/root/datasets/cifar10/cifar-10-batches-bin`，你可以在 `/path/to/automl/examples/nas/darts_cnn/darts_tf.yml` 文件里修改：

```
fully_train:                # fully_train阶段的参数配置
    dataset:                # 数据集配置
        type: Cifar10
        common:
            data_path: /root/datasets/cifar10/cifar-10-batches-bin
``` 









