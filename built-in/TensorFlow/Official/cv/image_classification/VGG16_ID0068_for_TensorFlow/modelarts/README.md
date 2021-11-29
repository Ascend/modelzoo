# VGG16-ModelZoo (图像分类/TensorFlow)

---
## 1.概述
VGG16是一个经典的目标分类网络。整个网络都使用（3x3）的卷积核和（2x2）最大池化层，既可以保证感受视野，又能够减少卷积层的参数。Ascend提供的VGG16是基于TensorFlow实现的版本。

## 2.训练
### 2.1.算法基本信息
- 任务类型: 图像分类
- 支持的框架引擎: Ascend-Powered-Engine-TF-1.15-python3.7-aarch64
- 算法输入:
    - obs数据集路径，下面放已转换好的tfrecord格式数据集。
- 算法输出:
    - 训练生成的ckpt模型

### 2.2.训练参数说明
名称|默认值|类型|是否必填|描述
---|---|---|---|---|
lr|0.01|float|True|初始学习率
iterations_per_loop|10|int|True|每次迭代的步数
batch_size|32|int|True|一次训练所抓取的数据样本数量
max_train_steps|100|int|False|最大训练步数，当max_epochs未填时有效
max_epochs|-|int|False|训练轮数
mode|train|str|False|训练模式，可选train/evaluate/train_and_evaluate
class_num|1000|int|False|数据集类别数（迁移学习必填）
restore_path|-|str|False|迁移学习预加载模型路径（迁移学习必填；若正常训练，则不需要上传预训练模型，创建训练任务时不配置该参数）
restore_exclude|['dense_2']|list|False|迁移学习中需要忽略的层


### 2.3.训练输出文件
训练完成后的输出文件如下
```
训练输出目录
  |- vgg16.log
  |- checkpoint
  |- events...
  |- graph.pbtxt
  |- model.ckpt-xxx
  |- ...
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
  |- model.ckpt-750600.data-00000-of-00001
  |- model.ckpt-750600.index
  |- model.ckpt-750600.meta   
```

### 3.2. 修改调优参数
目前迁移学习支持修改数据集类别，订阅算法创建训练任务，修改class_num和restore_path两个调优参数，其中：

- class_num为数据集类别数，根据用户实际数据集的分类填写的class_num为用户数据集类别。
- restore_path为对应预训练模型名称，默认在obs数据目录下。

调优参数示例如下。
```
class_num = 10
restore_path = model.ckpt-750600
```

### 3.3. 创建训练作业
指定数据存储位置、模型输出位置和作业日志路径，创建训练作业进行迁移学习。