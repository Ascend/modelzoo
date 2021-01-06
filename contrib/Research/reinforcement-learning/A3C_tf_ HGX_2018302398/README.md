# A3C

## 概述

迁移强化学习模型A3C到ascend910平台，将Ascend平台训练结果的精度和论文在cpu上的训练结果进行对比。

| 测试游戏 | 论文得分收敛值 | 论文训练时间 | Ascend得分收敛值 | Ascend训练时间 |
| -------- | -------------- | ------------ | ---------------- | -------------- |
| Pong     | 11.4           | 1 day        | 17.4             | 3 hour         |

## Requirement

* Tensorflow 1.15.0.
* Ascend910
* gym[atari] gym=0.10.5

## 代码路径解释

```shell
A3C
└─ 
  ├─README.md
  ├─model_pong 对pong游戏训练出的模型
  	├─checkpoint
  	├─model.ckpt.data-00000-of-00001
  	├─model.ckpt.index
  	├─model.ckpt.meta
  ├─log_pong 训练pong游戏模型时的日志
  	├─log.txt 日志
  	├─W_X 第X线程下的模型结构
  ├─Params.py 全局变量定义
  ├─AcNet.py 核心网络结构
  ├─model.py 模型结构
  ├─workerm.py 每一个线程下工作单元worker结构
  ├─env.py gym环境预处理
  ├─train.py 启动训练脚本
```

## 参数解释

--output_url 输出模型目录

--log_url 输出日志目录

--seed 环境随机种子

--env_name 测试游戏名称

--threads_num 线程数

--UPDATE_GLOBAL_ITER 更新全局网络步长

--MAX_GLOBAL_EP 总训练步长

--lr 学习率

--GAMMA 贡献递减率

--ENTROPY_BETA 熵损失参数

## 训练

```shell
python --env_name PongDeterministic-v4 --threads_num 16
```

