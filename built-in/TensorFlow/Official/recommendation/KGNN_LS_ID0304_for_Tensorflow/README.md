# KGNN-LS for Recommender Systems

This repository is the implementation of KGNN-LS ([arXiv](http://arxiv.org/abs/1905.04413)):

> Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems
> Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao, Wenjie Li, Zhongyuan Wang.  In Proceedings of The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019)

原始模型参考[[github链接](https://github.com/hwwang55/KGNN-LS)]，迁移训练代码到NPU Ascend 910

## 结果展示

|                   | 精度（eval auc） | 性能（s/step） |
| :---------------: | :--------------: | -------------- |
|       基线        |      85.00%      | 0.2            |
| NPU（Ascend 910） |      84.50%      | 0.15           |

## 快速启动
在src目录下执行main.py即可：

``````
python3 main.py --data_path=../data/restaurant
``````

也可以执行test目录下的归一化脚本：

```bash train_full_1p.sh --data_path=../data/restaurant```  或

```bash train_performance_1p.sh --data_path=../data/restaurant```

```
|-- data                  ----数据集目录
    |--restaurant     
        |--ratings_final.txt 
        |--kg_final.txt 
|-- src                   ----模型脚本目录
    |--aggregators.py
    |--data_loader.py	  ----数据导入脚本
    |--dataset.py	  	  ----数据处理脚本
    |--empirical_study.py
    |--main.py	  		  ----训练入口脚本
    |--model.py	  		  ----模型文件
    |--preprocess.py	  ----预处理文件
    |--train.py	  		  ----训练脚本
|-- README.md             ----使用前必读
|-- test                  ----NPU训练归一化shell
    |--env.sh
    |--launch.sh
    |--train_full_1p.sh
    |--train_performance_1p.sh
```

## Setup

Download this repository

## Requirements
* TensorFlow 1.15
* Ascend 910

## Dataset

注意：github已包含数据集，https://github.com/hwwang55/KGNN-LS/tree/master/data/restaurant

`Dianping-Food.zip`: containing the final rating file and the final KG file;

执行解压即可

`unzip Dianping-Food.zip`

## Run
``````
python3 main.py --data_path=../data/restaurant
``````

也可以执行归一化脚本：

```bash train_full_1p.sh --data_path=../data/restaurant```  或

```bash train_performance_1p.sh --data_path=../data/restaurant```

