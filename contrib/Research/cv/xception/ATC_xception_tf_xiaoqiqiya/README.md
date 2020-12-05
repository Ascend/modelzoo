### -  **XCEPTION** 


-  **概述** 
迁移Xception到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| Top-1 accuracy | 0.79 | 0.781  |

-  **Requirements** 
1. Tensorflow 1.15
2. Ascend910

-  **代码及路径解释** 


```
xception
└─ 
  ├─README.md
  ├─data 用于存放数据集
  	├─val.record
  	└─...
  ├─model 用于存放预训练模型
  	├─checkpoint
  	├─xception_model.ckpt.data-00000-of-00001
  	├─xception_model.index
  	├─xception_model.meta
  	└─...
  ├─save_model 用于存放经过fine_turn后的模型文件
  	├─checkpoint
  	├─xception_model.ckpt.data-00000-of-00001
  	├─xception_model.index
  	├─xception_model.meta
  	└─...
  ├─xception_model.py xception网络架构
  ├─run_xception.py 进行train和eval的一些逻辑操作
  ├─run.sh 模型的启动脚本，其中包含两种模式，一种是加载预训练模型继续训练，另一种是加载模型进行eval
```


