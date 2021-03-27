###   **dpn** 


###   **概述** 

迁移dpn到ascend910平台
将结果与gpu端进行比较

 |                | gpu   | ascend |
|----------------|------|--------|
| miou | 0.4376 | 0.4422  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
dpn
└─ 
  ├─README.md
  ├─data 用于存放训练和验证数据集 #obs://dpn-new/train_data/
  ├                             #obs://dpn-new/val_data/
  ├─model_save 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─dpn.ckpt.data-00000-of-00001
  	├─dpn.index
  	├─dpn.meta
  	└─...
  ├─DPN_model.py 定义dpn的模型架构
  ├─train_npu.py dpn训练文件
  ├─test_npu.py dpn测试文件
  ├─train_1p.sh 模型的启动脚本，自动从model文件夹中加载最后一次训练模型
  ├─test_1p.sh 模型的启动测试脚本
```
###   **数据集和模型** 

数据集 cvcdb
https://polyp.grand-challenge.org/site/Polyp/CVCClinicDB/


### 训练过程及结果
epoch=57
batch_size=8
lr=0.0001

训练预计13小时
