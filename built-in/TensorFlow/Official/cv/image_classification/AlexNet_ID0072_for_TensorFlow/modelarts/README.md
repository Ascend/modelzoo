# AlexNet-ModelZoo (图像分类/TensorFlow)

---
## 1.概述
AlexNet是一个经典的图像分类网络，AlexNet在2012年ImageNet竞赛中获得图像分类冠军，AlexNet中包含了几个比较新的技术点，也是首次在CNN中成功应用ReLU、Dropout和LRN等trick，同时AlexNet也使用了GPU进行加速运算。整个网络使用了1个11x11的卷积核、1个5x5的卷积核和3个3x3的卷积核，AlexNet全部使用最大池化，避免了平均池化的模糊化效果。Ascend提供的AlexNet是基于TensorFlow实现的版本。

## 2.训练
### 2.1.算法基本信息
- 任务类型: 图像分类
- 支持的框架引擎: Ascend-Powered-Engine-TF-1.15-python3.7-aarch64
- 算法输入:
    - obs数据集路径，下面放已转换好的tfrecord格式数据集
- 算法输出:
    - 训练生成的ckpt模型

### 2.2.训练参数说明
名称|默认值|类型|是否必填|描述
---|---|---|---|---|
max_epochs|150|int|True|训练轮数
lr|0.015|float|True|初始学习率
iterations_per_loop|100|int|True|每次迭代的步数
batch_size|256|int|True|一次训练所抓取的数据样本数量
freeze_pb|True|bool|False|训练结束是否将最后一个ckpt冻结为pb模型文件
num_classes|1000|int|False|数据集类别数（迁移学习必填）
restore_path|-|string|False|迁移学习预加载模型路径（迁移学习必填；若正常训练，则不需要上传预训练模型，创建训练任务时不配置该参数）

### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- alexnet_training.log
  |- checkpoint
  |- events...
  |- graph.pbtxt
  |- model.ckpt-xxx
  |- ...
  |- model.pb
  |- alexnet_tf_910.pb
```

## 3.迁移学习指导
### 3.1.上传预训练模型ckpt文件到obs数据目录，示例如下：
```
obs数据目录
  |- train-00000-of-01024
  |- train-00001-of-01024
  |- train-00002-of-01024
  |- ...
  |- validation-00000-of-00128
  |- validation-00001-of-00128
  |- validation-00002-of-00128
  |- ckpt_pretrained
        |- model.ckpt-187700.data-00000-of-00001
        |- model.ckpt-187700.index
        |- model.ckpt-187700.meta   
```

### 3.2. 修改调优参数
目前迁移学习支持修改数据集类别，订阅算法创建训练任务，修改num_classes和restore_path两个调优参数，其中：

- num_classes为数据集类别数，根据用户实际数据集的分类填写的num_classes为用户数据集类别。
- restore_path为对应预训练模型名称，默认在obs数据目录下的ckpt_pretrained目录下。

调优参数示例如下。
```
num_classes = 10
restore_path = model.ckpt-187700
```

### 3.3. 创建训练作业
指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。