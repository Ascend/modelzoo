# EfficientNet-B0-ModelZoo (图像分类/TensorFlow)

---
## 1.概述
EfficientNets是由Google发布的一系列图像分类网络，EfficientNet-B0是该系列网络中的轻量级基础网络。EfficientNets中的其他网络以EfficientNet-B0为基础，通过缩放生成。EfficientNet-B0网络的基本block结构中，同时使用了mobile inverted bottleneck block和squeeze-and-excitation block。Ascend提供的EfficientNet-B0是基于TensorFlow实现的版本。

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
train_steps|218750|int|True|总的训练步数
base_learning_rate|0.2|float|True|初始学习率
iterations_per_loop|625|int|True|每次迭代的步数
train_batch_size|256|int|True|一次训练所抓取的数据样本数量
mode|train|string|True|脚本运行模式，可选模式包括train、eval或train_and_eval，默认train
steps_per_eval|31250|int|False|当mode=train_and_eval时，需要传递的参数，表示每隔多少步验证一次
freeze_pb|True|bool|False|训练结束是否将最后一个ckpt冻结为pb模型文件
num_label_classes|1000|int|False|数据集类别数（迁移学习必填）
restore_path|-|string|False|迁移学习预加载模型路径（迁移学习必填；若正常训练，则不需要上传预训练模型，创建训练任务时不配置该参数）

### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- train_log.log
  |- checkpoint
  |- events...
  |- graph.pbtxt
  |- model.ckpt-xxx
  |- ...
  |- efficientnet-b0_tf.pb
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
restore_path = model.ckpt-218750
```

### 3.3. 创建训练作业
指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。