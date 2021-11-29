# DenseNet-121-ModelZoo (图像分类/TensorFlow)

---
## 1.概述
DenseNet-121是一个经典的图像分类网络，主要特点是采用各层两两相互连接的Dense Block结构。为了提升模型的效率，减少参数，采用BN-ReLU-Conv（1*1）-BN-ReLU-Conv（3*3）的bottleneck layer，并用1*1的Conv将Dense Block内各层输入通道数限制为4k（k为各层的输出通道数）。DenseNet能有效缓解梯度消失，促进特征传递和复用。

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
lr|0.1|float|True|初始学习率
iterations_per_loop|10|int|True|每次迭代的步数
batch_size|32|int|True|一次训练所抓取的数据样本数量
freeze_pb|True|bool|False|训练结束是否将最后一个ckpt冻结为pb模型文件
num_classes|1000|int|False|数据集类别数（迁移学习必填）
restore_path|-|string|False|迁移学习预加载模型路径（迁移学习必填；若正常训练，则不需要上传预训练模型，创建训练任务时不配置该参数）

### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- densenet121_training.log
  |- checkpoint
  |- events...
  |- graph.pbtxt
  |- model.ckpt-xxx
  |- ...
  |- model.pb
  |- densenet121_tf_910.pb
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
        |- model.ckpt-746000.data-00000-of-00001
        |- model.ckpt-746000.index
        |- model.ckpt-746000.meta   
```

### 3.2. 修改调优参数
目前迁移学习支持修改数据集类别，订阅算法创建训练任务，修改num_classes和restore_path两个调优参数，其中：

- num_classes为数据集类别数，根据用户实际数据集的分类填写的num_classes为用户数据集类别。
- restore_path为对应预训练模型名称，默认在obs数据目录下的ckpt_pretrained目录下。

调优参数示例如下。
```
num_classes = 10
restore_path = model.ckpt-746000
```

### 3.3. 创建训练作业
指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。