###   **Double_UNet** 


###   **概述** 

迁移Double_UNet到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| miou | 0.8611 | 0.8610  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
Double_UNet
└─ 
  ├─README.md
  ├─dataset 用于存放训练和验证数据集 #obs://double-unet/dataset/
  	├─cvcdb_train.h5
  	└─cvcdb_val.h5
  ├─model 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─model.ckpt.data-00000-of-00001
  	├─model.index
  	├─model.meta
  	└─...
  ├─Dataset.py 定义数据集的生成和读取操作
  ├─DoubleUnet.py 定义DoubleUnet的模型架构
  ├─layers.py 封装基础的层
  ├─train.sh 模型的启动脚本，自动从model文件夹中加载最后一次训练模型
  ├─test.sh 模型的启动测试脚本
```
###   **数据集和模型** 

数据集 cvcdb
https://polyp.grand-challenge.org/site/Polyp/CVCClinicDB/


### 训练过程及结果
epoch=300
batch_size=16
lr=1e-5