## deeplab v2

## 概述
迁移deeplabv2到ascend910平台上使用NPU运行，并将结果与原论文进行对比
| Accuracy | Paper | Ours  |
|----------|-------|-------|
| mIoU     | 0.715 | 0.744 |

## Requirements

1.Tensorflow 1.15

2.Ascend910

## 代码及路径解释
```
deeplabv2
└─
  ├─README.md
  ├─LICENSE  
  ├─dataset 用于存放训练集，验证集，测试集的标签
  	├─train.txt
        ├─val.txt
        ├─test.txt
  	└─...
  ├─testcase 用于存放自测试用例标签
  	├─train.txt
        ├─val.txt
        ├─test.txt
  	└─...
  ├─model 用于存放预训练模型 obs//deeplab-zjw/dataset/pretraind_model/deeplab_resnet.ckpt
  	├─deeplab_resnet.ckpt
  	└─...
  ├─trained_model 用于存放训练后的模型文件
  	├─checkpoint
        ├─image_reader.py
        ├─label_utils.py
        ├─write_to_log.py 
  	└─...
  ├─utils 用于存放数据预处理文件
  	├─__init__.py
        ├─model.ckpt-2000.data-00000-of-00001
        ├─model.ckpt-2000.index
        ├─model.ckpt-2000.meta
  	└─...
  ├─main.py 执行主函数代码
  ├─model.py 定义模型train，eval过程的逻辑操作
  ├─network.py 搭建网络结构
  
```
## 数据集和预训练模型

数据集：

预训练模型：

## 训练过程及结果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1226/215613_87d9d711_8310380.png "屏幕截图.png")

## 执行训练

python main.py 

加载预训练模型后，共计耗时2个小时左右

## 执行验证

python main.py --option test

加载NPU训练的模型，共计耗时一个半小时左右


