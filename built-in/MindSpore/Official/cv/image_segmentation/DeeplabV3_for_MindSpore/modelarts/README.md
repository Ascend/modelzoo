# DeepLabv3-ModelZoo (语义分割/MindSpore)

---
## 1.概述
DeepLab是一系列的图像语义分割模型，和以前的版本相比，DeepLabv3有了很大的改进。DeepLabv3的两个关键点：
- multi-grid空洞卷积可以更好地处理不同尺度的分割物体。
- 改进的ASPP能够利用图像级的特征，去捕获长距离的信息。

## 2.训练
### 2.1.算法基本信息
- 任务类型: 语义分割分类
- 支持的框架引擎: Ascend-Powered-Engine-Mindspore-1.1.1-python3.7-aarch64
- 算法输入:
    - obs数据集路径
- 算法输出:
    - 训练生成的ckpt模型

### 2.2.训练参数说明
名称|默认值|类型|是否必填|描述
---|---|---|---|---|
train_epochs|200|int|True|总轮次数
batch_size|32|int|True|输入张量的批次大小
crop_size|513|int|True|裁剪大小
base_lr|0.015|float|True|初始学习率
lr_type|cos|str|False|用于生成学习率的衰减模式
lr_decay_step|40000|int|False|学习率衰减步数
num_classes|21|int|True|类别数
model|deeplab_v3_s16|str|True|选择模型，可选deeplab_v3_s16/deeplab_v3_s8
min_scale|0.5|float|False|数据增强的最小尺度
max_scale|2.0|float|False|数据增强的最大尺度
ignore_label|255|int|False|忽略标签
ckpt_pre_trained|-|str|False|加载预训练检查点的路径
save_steps|1500|int|False|用于保存的迭代间隙
keep_checkpoint_max|200|int|False|用于保存的最大检查点

### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- deeplab_v3_s16-graph.meta
  |- deeplab_v3_s16-xxx_xxx.ckpt
  |- ...
```

## 3.迁移学习指导
### 3.1.上传预训练模型ckpt文件到obs数据目录
```
obs数据目录
  |- eval
  |- test
  |- train
       |- deeplab_v3_s16-xxx_xxx.ckpt
       |- ...
```

### 3.2. 修改调优参数
目前迁移学习支持修改数据集类别，订阅算法创建训练任务，class_num和ckpt_pre_trained两个调优参数

调优参数示例如下。
```
class_num = 10
pre_trained = deeplab_v3_s16-xxx_xxx.ckpt
```

### 3.3. 创建训练作业
指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。