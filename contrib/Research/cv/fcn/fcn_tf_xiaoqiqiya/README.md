###   **fcn** 


###   **概述** 

迁移fcn到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| miou | 0.627 | 0.6259  |

###  Requirements

1. Tensorflow 1.15
2. Ascend910

###   **代码及路径解释** 



```
Double_UNet
└─ 
  ├─README.md
  ├─data 用于存放训练和验证数据集 #obs://fcn-8s/dataset/
  	├─val.tfrecords
  	└─train1.tfrecords
  ├─model 用于存放经过fine_turn前的模型文件#obs://fcn-8s/fcn_pretrain/
  	├─checkpoint
  	├─fcn.ckpt.data-00000-of-00001
  	├─fcn.index
  	├─fcn.meta
  	└─...
  ├─model_save 用于存放经过fine_turn后的模型文件
  ├─fcn_pretrain.py 定义fcn的模型架构
  ├─train_npu.py train文件
  ├─fcn_pretrain.py 定义fcn的模型架构
  ├─test_npu.py
  ├─train.sh 模型的启动脚本，自动从model文件夹中加载最后一次训练模型
  ├─test.sh 模型的启动测试脚本

```
###   **数据集和模型** 

数据集 voc2011
https://pjreddie.com/projects/pascal-voc-dataset-mirror/


### 训练过程及结果
epoch=9
batch_size=32
lr=1e-5

 **
offline_inference**
[offline_inference](https://gitee.com/xiaoqiqiyaya/modelzoo/tree/master/contrib/Research/cv/fcn/fcn_tf_xiaoqiqiya/offline_inference) 