# ResNet50-ModelZoo (图像分类/MindSpore)

---
## 1.概述
ResNet是ImageNet竞赛中分类问题效果比较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。

该算法提供的是50-layer的ResNet网络的V1.5版本，V1.5版本的ResNet网络与V1版本的区别在于，在bottleneck模块中，V1版本是在第一个1x1卷积层中设置stride=2，而V1.5是在3x3卷积层中设置stride=2。

## 2.训练
### 2.1.算法基本信息
- 任务类型: 图像分类
- 支持的框架引擎: Ascend-Powered-Engine-Mindspore-1.1.1-python3.7-aarch64
- 算法输入:
    - obs数据集路径
- 算法输出:
    - 训练生成的ckpt模型

### 2.2.训练参数说明
名称|默认值|类型|是否必填|描述
---|---|---|---|---|
epoch_size|90|int|True|训练轮数
batch_size|256|int|True|一次训练所抓取的数据样本数量
class_num|1001|int|True|数据集类数
pre_trained|-|string|False|迁移学习预加载模型路径（迁移学习必填；若正常训练，则不需要上传预训练模型，创建训练任务时不配置该参数）

### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- resnet-graph.meta
  |- resnet-5_136.ckpt
  |- resnet-10_136.ckpt
  |- ...
```

## 3.迁移学习指导
### 3.1.上传预训练模型ckpt文件到obs数据目录
```
obs数据目录
  |- eval
  |- test
  |- train
       |- resnet-90_625.ckpt
       |- ...
```

### 3.2. 修改调优参数
目前迁移学习支持修改数据集类别，订阅算法创建训练任务，class_num和pre_trained两个调优参数

调优参数示例如下。
```
class_num = 10
pre_trained = /path/to/pre_trained.ckpt
```

### 3.3. 创建训练作业
指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。