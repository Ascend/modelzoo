# UNet-ModelZoo (语义分割/MindSpore)

---
## 1.概述
U-Net是一个经典的图像分割网络，在多种图像分割任务性能突出。U-Net结构由收缩部分和扩张部分组成，在网络中形成瓶颈结构（bottleneck），收缩部分由卷积和池化组成使图像降采样，经过瓶颈后，通过卷积和上采样的扩张部分重建图像。在收缩部分和扩张部分添加跳连，帮助训练梯度回传。U-Net根据通道数、网络层数、上采样类型等不同有不同的变种，Ascend提供的U-Net是基于MindSpore实现的版本，和论文中原始的UNet网络结构有如下区别：
- 提升网络整体的通用性，允许修改网络input shape
- 移除上采样中的固定大小的Crop操作
- 移除数据预处理阶段固定大小的Crop操作
- 使用pad_mode=same的卷积替换pad_mode=valid的卷积

## 2.训练
### 2.1.算法基本信息
- 任务类型: 语义分割
- 支持的框架引擎: Ascend-Powered-Engine-Mindspore-1.1.1-python3.7-aarch64
- 算法输入:
    - obs数据集路径，下面放2018 Data Science Bowl或ISBI-2012格式的数据集
- 算法输出:
    - 训练生成的ckpt模型

### 2.2.训练参数说明
名称|默认值|类型|是否必填|描述
---|---|---|---|---|
dataset|Cell_nuclei|string|True|数据集类型，可选填Cell_nuclei、ISBI2012
img_size|[96, 96]|list|True|网络输入图像大小
lr|0.0003|float|True|学习率
epochs|200|int|True|训练轮次
repeat|10|int|True|每一遍epoch重复数据集的次数
batchsize|16|int|True|训练批次大小
num_classes|2|int|True|数据集类数
num_channels|3|int|True|输入图像通道数

### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- ckpt_unet_simple_adam-11_335.ckpt
  |- ckpt_unet_simple_adam-12_335.ckpt
  |- ...
  |- ckpt_unet_simple_adam-20_335.ckpt
  |- ckpt_unet_simple_adam-graph.meta
```
