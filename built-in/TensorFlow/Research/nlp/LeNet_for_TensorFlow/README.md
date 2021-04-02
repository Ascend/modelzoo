# 目录
-   [交付件基本信息](#交付件基本信息)
-   [概述](#概述)
-   [快速上手](#快速上手)
-   [迁移学习指导](#迁移学习指导)
-   [高级参考](#高级参考)

# [交付件信息](#交付件信息)

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Classification**

**版本（Version）：1.1**

**修改时间（Modified）：2021.4.2**

_**大小（Size）**_**: 16K**

**框架（Framework）：_Tensorflow 1.15.0_**

**模型格式（Model Format）：_ckpt_**_

**精度（Precision）：Mixed**

**处理器（Processor）：_昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：给予Tensorflow框架的图片中手写0~9数字分类**

# [概述](#概述)



## 简述

LeNet用于图片中手写数字的识别分类
开源代码网址：https://github.com/Jackpopc/aiLearnNotes/blob/master/computer_vision/LeNet.py 



## 默认配置

-   训练超参（单卡）：
    - Batch size： 64
    - Train epoch: 5
    - Train step: 1000


## 支持特性

- 混合精度：
  开启混合精度，训练时对于既支持float32精度也支持float16精度的算子或节点，按照内置的优化策略，自动将算子降低精度到float16

- loss scale：

  在混合精度计算中使用float16数据格式数据动态范围降低，造成梯度计算出现浮点溢出，会导致部分参数更新失败。为了保证模型训练在混合精度训练过程中收敛，需要配置Loss Scaling的方法。
  Loss Scaling方法通过在前向计算所得的loss乘以loss scale系数S，起到在反向梯度计算过程中达到放大梯度的作用，从而最大程度规避浮点计算中较小梯度值无法用FP16表达而出现的溢出问题。在参数梯度聚合之后以及优化器更新参数之前，将聚合后的参数梯度值除以loss scale系数S还原。



## 混合精度训练

混合精度训练方法是通过混合使用float16和float32数据类型来加速深度神经网络训练的过程，并减少内存使用和存取，从而可以训练更大的神经网络。同时又能基本保持使用float32训练所能达到的网络精度。当前昇腾AI处理器支持如下几种训练精度模型，用户可以在训练中设置。

- allow_fp32_to_fp16: 优先保持原图精度，当算子不支持float32数据类型时，直接降低到float16
- force_fp16: 当算子既支持float16又支持float32数据类型时，强制选择float16.
- must_keep_origin_dtype: 保持原图精度。
- allow_mix_precision: 自动混合精度。针对全网中float32数据类型的算子，按照按照内置的优化策略，自动将算子降低精度到float16。

## 开启混合精度

拉起训练的命令行中，增加参数“precision_mode”, 例如：
sh +x train_full_1p.sh --data_path="../MNIST" --precision_mode=allow_mix_precision

# [训练环境准备](#训练环境准备)

1.  硬件环境准备请参见[各硬件产品文档](https://ascend.huawei.com/#/document?tag=develoger)。需要在硬件设备上安装固件与驱动。
2.  安装Ascend910软件包
3.  安装python3.7.5、Tensorflow 1.15.0

# [快速上手](#快速上手)

## 数据集准备

请用户自行准备数据集"MNIST", 并将数据集放在脚本的根目录下。



## 模型训练

1. 下载训练脚本

2. 下载数据集

3. 上传源代码到训练机器并解压，同时上传数据集到代码目录

4. 登陆到训练机器，进入test目录，执行如下命令：
   
   sh +x train_full_1p.sh --data_path="../MNIST"
 
 5. 验证。
 
    打开 ./test/output下的训练统计文件，vim output/performance_precision.txt, 可以查看保存的训练精度与性能数据。

    精度基线：0.975
    性能：0.0021 sec/step


# [迁移学习指导](#迁移学习指导)



# [高级参考](#高级参考)



## 脚本和示例代码



## 脚本参数



## 训练过程



## 推理/验证过程


