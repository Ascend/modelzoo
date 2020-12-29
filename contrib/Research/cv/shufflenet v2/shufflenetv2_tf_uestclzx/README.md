## shufflenetv2

## 概述

迁移shufflenetv2到ascend910平台上使用NPU训练，并将结果与原论文进行对比

| Accuracy | Paper | Ours  |
|----------|-------|-------|
| Top-1    | 0.603 | 0.615 |

## 代码及路径解释

```
shufflenetv2
└─
  ├─README.md
  ├─LICENSE  
  ├─dataset 用于存放训练集，验证集
   ├─train
     ├─shard-0000.tfrecords
   ├─val
     ├─shard-0000.tfrecords
   └─...
  ├─pretrained_model 用于存放预训练模型 
   ├─checkpoint
   ├─model.ckpt-1050924.data-00000-of-00001
   ├─model.ckpt-1050924.meta
   ├─model.ckpt-1050924.index
   └─...
  ├─trained_model 用于存放生成的模型，用于验证
   ├─checkpoint
   ├─model.ckpt-1601408.data-00000-of-00001
   ├─model.ckpt-1601408.meta
   ├─model.ckpt-1601408.index
   └─...
  ├─train.py 执行训练
  ├─eval.py 执行验证
  ├─architecture.py 搭建网络结构
  ├─model.py 定义图操作
  ├─layers.py 定义搭建模型所需函数
  ├─input_pipline.py 数据处理
```
## 数据集合预训练模型

数据集：ImageNet12 桶地址：obs://zjw-shufflenet/data

预训练模型：checkpoint 桶地址：obs://zjw-shufflenet/data/pretrained-model

![桶配置](https://images.gitee.com/uploads/images/2020/1229/211943_07b96269_8511959.png "屏幕截图.png")
## 训练过程及打印结果
![输入图片说明](https://images.gitee.com/uploads/images/2020/1229/170252_b439bc45_8310380.png "屏幕截图.png")

## 执行训练

```
python train.py

```
预计耗时30h

## 执行验证

```
python eval.py
```
预计耗时30min