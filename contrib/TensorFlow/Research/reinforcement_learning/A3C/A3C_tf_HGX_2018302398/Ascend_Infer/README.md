# A3C推理

## 推理效果

| 测试游戏 | GPU推理pb模型速度(NVIDIA T4) | npu推理om模型速度(Ascend310) | 推理得分 |
| -------- | ---------------------------- | ---------------------------- | -------- |
| Pong     | 2.63 s                       | 2.21 s                       | 21       |
| Alien    |                              |                              |          |

## Requirement

* python3.7
* Tensorflow 1.15.0
* gym=0.10.5
* atari-py
* Ascend310 + CANN(20.1.alpha001)

## 代码路径解释

```bash
├─Ascend_Infer Ascend310平台上推理脚本
  	├─A3C_Inferance.py	推理脚本
  	├─acl_model.py		定义模型类，完成模型推理过程中资源管理
  	├─constants.py		常量定义
  	├─utils.py			常用操作定义
  	├─envs.py			gym环境类的重载
```

## 推理流程描述

A3C_Inferance.py 脚本对指定om模型调用Ascend推理资源进行推理

## 推理脚本：A3C_Inferance.py 

#### 参数

|   参数名    |     参数作用     |
| :---------: | :--------------: |
| input_model | 输入om模型的路径 |
|  game_name  | 待测游戏环境名称 |
|  test_num   |     测试轮数     |

#### 示例

```bash
python A3C_Inferance.py --input_model ../om_model/a3c_pong_model.om --game_name PongDeterministic-v4 --test_num 10
```

