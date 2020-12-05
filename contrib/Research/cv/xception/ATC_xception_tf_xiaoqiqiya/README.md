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
albert
└─ 
  ├─README.md
  ├─output_base_v2 基于squadv2微调过的albert base模型路径
  	├─checkpoint
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  	└─...
  ├─output_large_v2 基于squadv2微调过的albert base模型路径
  	├─checkpoint
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  	└─...
  ├─albert_base_v2 albert base的预训练模型
  	├─30k-clean.model
  	├─30k-clean.vocab
  	├─albert_config.json
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta

  ├─albert_large_v2 albert large的预训练模型
  	├─30k-clean.model
  	├─30k-clean.vocab
  	├─albert_config.json
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  ├─squad_v2 存放数据目录
  	├─train-v2.0.json 数据源文件
  	├─dev-v2.0.json 数据源文件
  	├─train.tfrecord 根据train-v2.0.json生成的文件
  	├─dev.tfrecord 根据dev-v2.0.json生成的文件
  	├─pred_left_file.pkl 根据dev-v2.0.json生成的文件

  ├─squad2_base.sh albert base的启动脚本
  ├─squad2_large.sh albert large的启动脚本
```


