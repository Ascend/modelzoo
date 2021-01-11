## shufflenetv1

## 概述

迁移shufflenetv1到ascend910平台上使用NPU训练，并将结果与原论文进行对比

| Accuracy | Paper | Ours  |
|----------|-------|-------|
| Top-1    | 0.568 | 0.565 |

## 代码及路径解释

```
shufflenetv
└─
  ├─README.md
  ├─LICENSE  
  ├─dataset 用于存放训练集，验证集
   ├─train
     ├─shard-0000.tfrecords
   ├─val
     ├─shard-0000.tfrecords
   └─...
  ├─model 用于存放预训练模型和生成的模型，用于验证
   ├─checkpoint                                #预训练模型
   ├─model.ckpt-1254431.data-00000-of-00001
   ├─model.ckpt-1254431.meta
   ├─model.ckpt-1254431.index
   ├─...
   ├─model.ckpt-1281121.data-00000-of-00001    #生成的模型
   ├─model.ckpt-1281121.meta
   ├─model.ckpt-1281121.index
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

预训练模型：checkpoint 桶地址：obs://modelarts-zjw/pretrained_model


## 训练过程及验证结果
![输入图片说明](https://images.gitee.com/uploads/images/2021/0111/171048_48ebe3c1_8511959.png "屏幕截图.png")

## 执行训练

```
python train.py

```
预计耗时3h

## 执行验证

```
python eval.py
```
预计耗时30min