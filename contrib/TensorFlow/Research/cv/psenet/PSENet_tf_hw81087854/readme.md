# PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network
原始模型参考[github链接](https://github.com/liuheng92/tensorflow_PSENet),迁移训练代码到NPU

## Requirements
- Tensorflow 1.15.0.
- Ascend910
- 其他依赖参考requirements.txt
- 数据集，下面有百度网盘下载链接，提取码1234

## 代码路径解释
```shell
.
├── checkpoint        ----存放训练ckpt的路径
├── eval.py           ----推理入口py     
├── eval.sh           ----推理shell，计算icdar2015测试集的精度、召回率、F1 Score
├── evaluation        ----精度计算相关的py，新增
├── LICENSE
├── nets              ----网络模型定义，包含backbone
│   ├── __init__.py
│   ├── model.py
│   ├── __pycache__
│   └── resnet
├── npu_train.py      ----NPU训练
├── ocr               ----数据集存放目录
│   ├── ch4_test_images  --test图片
│   └── icdar2015        --train图片
├── pretrain_model    ----backbone
├── pse               ----后处理PSE代码
│   ├── include
│   ├── __init__.py
│   ├── Makefile
│   ├── pse.cpp
│   ├── pse.so
│   └── __pycache__
├── readme.md
├── train_npu.sh     ----NPU训练入口shell
├── train.py         ----GPU训练
└── utils            ----数据集读取和预处理
    ├── data_provider
    ├── __init__.py
    ├── __pycache__
    └── utils_tool.py
```

## 准备数据和Backbone模型
Icdar2015、Icdar2017可以去官网下载，或者直接从百度网盘里面获取，Backbone使用Resnet50_v1 [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) [BaiduYun link，提取码1234](https://pan.baidu.com/s/1gh8q0WWoqWXHHtIumUG_Mg) 

存放目录参考上面的解释。

## 一些说明
1、原始Github链接中，作者给出的预训练模型基于Icdar2015+Icdar2017数据集训练，Icdar2015测试集评估，
![输入图片说明](https://images.gitee.com/uploads/images/2021/0219/235136_f88bf050_8432352.png "屏幕截图.png")
精度数据：
|     | Precision | Recall | Hmean |
|-----|-----------|--------|-------|
| GPU | 0.766     | 0.677  | 0.719 |

2、给出的训练超参也是基于预训练模型进行Finetune的超参：
```
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 32, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './resnet_train/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
```
3、本次实现，重新调整超参，使用Resnet50_v1预训练模型作为BackBone，使用Icdar2015和Icdar2015+Icdar2017数据集重新进行训练。

