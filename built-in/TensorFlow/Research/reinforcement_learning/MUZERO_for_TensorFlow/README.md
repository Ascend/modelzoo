# MUZERO_for_TensorFlow

## 目录

* [概述](#概述)
* [要求](#要求)
* [默认配置](#默认配置)
* [快速上手](#快速上手)
  * [配置docker镜像](#配置docker镜像)
  * [关键配置修改](#关键配置修改)
  * [运行示例](#运行示例)
    * [训练](#训练)
* [高级](#高级)
  * [脚本参数](#脚本参数) 
  * [仿真器介绍](#仿真器介绍) 

## 概述

MUZERO（Deep Deterministic Policy Gradient）是deepmind提出的最新的基于mcts的强化学习方法，本代码采用MUZERO算法学习atari游戏中的pong游戏策略。Ascend提供的MUZERO是基于TensorFlow实现的版本。

参考论文：Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.arXiv preprint arXiv:1911.08265

参考实现：https://github.com/werner-duvaud/muzero-general

## 要求

- 安装有昇腾AI处理器的硬件环境
 

## 默认配置

- 网络结构

  actor学习率：0.0003

  优化器：Adam

  单卡batchsize：320

  buffer size：10000

  折扣系数gamma：0.99

  每个epoch最大step：200

- 状态空间：（以atari的pong为例）

  游戏返回的状态的（216，160，3）的图像信息，代码会RGB的图像信息转换为（84，84，1）的灰度信息，然后将连续四帧组合在一起生成shape为（84，84，4）的状态信息。

- 动作空间：

  参考atari中pong的定义，为六维离散动作。

- 训练数据获取：

  MUZERO算法无需准备训练数据集，通过与gym仿真器实时交互获取训练数据。

## 快速上手

### 配置docker镜像

1. 确认环境是否已导入版本镜像

```
[root@localhost package_1110]# docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu_arm          18.04               d663ce295a44        4 seconds ago       3.92GB
```

2. 启动docker镜像

```
docker run -it --network=host --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/log/npu/conf/slog:/var/log/npu/conf/slog -v /var/log/npu/profiling/container/0:/var/log/npu/profiling/ -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /root/ModelZoo_MUZERO_TF_NOAH/:/home/code -w /home --privileged=true --name=muzero_container_name ubuntu_arm:18.04 /bin/bash
```

3. 配置网络相关代理

```
export http_proxy=http://user:passwd@172.18.32.221:8080
export https_proxy=https://user:passwd@172.18.32.221:8080
export GIT_SSL_NO_VERIFY=1
```

4. 配置pip源

```
cd ~
vi .pip/pip.config
[global]
trusted-host=10.93.238.51
index-url=http://10.93.238.51/pypi/simple/
```

5. 容器中安装opencv

```
apt-get update
apt-get install cmake
cd /home/code/opencv4.2.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/opencv ..

make -j8
make install

sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sh -c 'echo "/usr/local/opencv/lib" >> /etc/ld.so.conf.d/opencv.conf'
```

建立软连接

```
find /usr/local/ -type f -name "cv2*.so"
```

得到查找结果 path

```
cd /usr/local/lib/python3.7/site-packages
ln -s path cv2.so
```

6.安装刑天

```
cd /home/code
pip3 install -e .
```

### 关键配置修改

 启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


### 运行示例

#### 训练

启动训练脚本：

```
cd /home/code/
source env.sh
python3 xt/main.py -f examples/default_cases/muzero_pong_new.yaml
```

查看训练结果:

```
cd ~
tensorboard --logdir=xt_archive
```

## 高级

### 脚本参数

配置文件为 `/ModelZoo_MUZERO_TF_NOAH/rl/examples/default_cases/muzero_pong_new.yaml`

```
alg_para:
  alg_name: MuzeroNew                 # 算法名称，与代码对应
  alg_config: {
    "train_per_checkpoint": 10,      # 每训练n步，保存一次log信息
    "prepare_times_per_train": 1     # 收到多少数据触发一次训练
    }

env_para:
  env_name: AtariEnv                  # 仿真环境名称，与代码对应
  env_info: {'name': PongNoFrameskip-v4, vision': False}      # 指定atari中的pong环境

agent_para:
  agent_name: MuzeroPongNew           # 探索智能体名称，与代码对应
  agent_num : 1                       # 探索智能体数量
  agent_config: {
    'max_steps': 200,                 # 每个epoch 最大步长
    'complete_step': 5000000,         # 整个训练最大步长
    'NUM_SIMULATIONS': 50             # 蒙卡仿真次数
    }

model_para:
  actor:
    model_name: MuzeroPongTest        # actor 网络名称，与代码对应
    state_dim: [84, 84, 4]            # 状态维度
    action_dim: 6                     # 动作维度
    init_weights: /home/ModelZoo_MUZERO_TF_NOAH/actor_01662.h5 # 预加载模型

  
env_num: 30                           # 并行采样环境数量
node_config: [["127.0.0.1", "username", "passwd"]]
```

### 仿真器介绍

Gym是OpenAI开源的一个用于开发和测试强化学习算法的标准仿真库而atari是gym下面的一个游戏库。该脚本训练使用atari中的pong游戏，在脚本的配置文件 `/ModelZoo_MUZERO_TF_NOAH/rl/examples/default_cases/muzero_pong_new.yaml`中可以方便灵活的使用gym中的其他仿真环境。

修改仿真环境名称及仿真环境对应状态与动作维度即可：

```
env_para:
  env_name: AtariEnv                  # 仿真环境名称，与代码对应
  env_info: {'name': PongNoFrameskip-v4, vision': False}      # 指定atari中的pong环境

model_para:
  actor:
    model_name: MuzeroPongTest        # actor 网络名称，与代码对应
    state_dim: [84, 84, 4]            # 状态维度
    action_dim: 6                     # 动作维度
    init_weights: /home/ModelZoo_MUZERO_TF_NOAH/actor_01662.h5 # 预加载模型
```
