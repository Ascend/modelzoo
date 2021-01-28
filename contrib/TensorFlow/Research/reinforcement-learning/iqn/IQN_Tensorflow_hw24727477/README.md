# IQN
## 概述
IQN论文的作者没有将代码开源，本代码改自github上个人实现的torch版本
得到的结果和论文的对比，
| Game | Return |
| :-----|:----: |
| Pong(Ascend) | **21.0**
| Pong(论文) | 21.0 
| Boxing(Ascend) | 99.7 
| Boxing(论文) | **99.8** 

## Requirements
- Tensorflow 1.15
- Ascend910
- Gym 0.10.9
- atari-py
- opencv-python

## 代码路径解释

```shell
IQN
└─ 
  ├─README.md
  ├─IQN_tf.py 模型的定义，以及训练，评估过程
  ├─monitor.py 
  ├─replay_memory.py 经验回放定义 
  ├─wrappers.pypy 并行环境的定义
```
---

## 准备数据和模型
所有数据均来自gym atari游戏库  
训练好的模型(Pong游戏)：链接：https://pan.baidu.com/s/1UWmBBe0sXV6DGtew9nwRxw 
提取码：frfz 



## 参数解释 
	--games = Pong (atari游戏的名称，如Pong，Breakout，Boxing等) 
	--load = 0 （0为重新训练模型，1为加载训练过的模型）
	--mode = train （train为训练，evaluate为评估）
	--loss_scale = 1024 (loss_scale系数，可适当调整)
	--train_url  模型保存路径
	--TARGET_PERLACE_ITER = 100 （目标网络更新的间隔步长）
	--MEMORY_CAPACITY = 1e+5 （经验回放的容量）
	--N_QUANT=64 (分位数数量)
	--N_ENVS=32 (并行环境数)
	--STEP_NUM = 4e+7 (训练步长，设置数量由游戏决定)
	--BATCH_SIZE=32
	--LR = 1e-4                             
	--SAVE_FREQ = 1e+3 (模型保存的间隔)

## 训练示例
	重新训练：
	python3 IQN_tf.py --mode=train --game=Pong --load=0 --train_url=./model_save
	在之前的基础上继续训练：
	python3 IQN_tf.py --mode=train --game=Pong --load=1 --train_url=./model_save 

## 评估示例
	python3 IQN_tf.py --mode=evaluate --game=Pong --train_url=./model_save

## 说明
1、如果想要加载训练的模型进行评估或者继续训练，需要把模型ckpt文件放在train_url参数的路径下

2、开启混合精度后，部分计算的精度被调整为fp16，导致在更新参数时出现梯度下溢出，进而导致网络发散，因此必须加入loss_scale来解决这个问题。