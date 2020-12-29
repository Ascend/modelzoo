### deeplab v2

### 概述
迁移deeplabv2到ascend910平台上使用NPU运行，并将结果与原论文进行对比
| Accuracy | Paper | Ours  |
|----------|-------|-------|
| mIoU     | 0.715 | 0.744 |

### Requirements

1.Tensorflow 1.15

2.Ascend910

### 代码及路径解释
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
  ├─img_data 用于存放图片数据集 obs//deeplab-zjw/dataset/
   ├─SegmentationClassAug
   ├─SegmentationClass
   ├─JPEGImages
   └─...
  ├─pretrained_model 用于存放预训练模型 obs//deeplab-zjw/dataset/pretraind_model/deeplab_resnet.ckpt
   ├─deeplab_resnet.ckpt
   └─...
  ├─utils 用于存放训练后的模型文件
   ├─checkpoint
   ├─image_reader.py
   ├─label_utils.py
   ├─write_to_log.py 
   └─...
  ├─trained_model 用于存放数据预处理文件
   ├─__init__.py
   ├─model.ckpt-2000.data-00000-of-00001
   ├─model.ckpt-2000.index
   ├─model.ckpt-2000.meta
   └─...
  ├─main.py 执行主函数代码
  ├─model.py 定义模型train，eval过程的逻辑操作
  ├─network.py 搭建网络结构
  
```
### 数据集和预训练模型：

数据集：PASCAL VOC 2012 dataset. 链接：http://host.robots.ox.ac.uk/pascal/VOC/

预训练模型：deeplab_resnet.ckpt 链接：https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU

### 训练过程及结果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1226/215613_87d9d711_8310380.png "屏幕截图.png")

### 执行训练

python main.py 

加载预训练模型后，共计耗时2个小时左右

主要参数注释：
```
num_steps: how many iterations to train

save_interval: how many steps to save the model

random_seed: random seed for tensorflow

weight_decay: l2 regularization parameter

learning_rate: initial learning rate

power: parameter for poly learning rate

momentum: momentum

encoder_name: name of pre-trained model, res101, res50 or deeplab

pretrain_file: the initial pre-trained model file for transfer learning

data_list: training data list file
```

### 执行验证

python main.py --option test

加载NPU训练的模型，共计耗时一个半小时左右

主要参数注释：
```
valid_step: checkpoint number for testing/validation

valid_num_steps: = number of testing/validation samples

valid_data_list: testing/validation data list file
```



