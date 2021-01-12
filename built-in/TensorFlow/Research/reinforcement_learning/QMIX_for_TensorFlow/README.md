# QMIX_for_TensorFlow

## 目录

* [概述](#概述)
* [要求](#要求)
* [默认配置](#默认配置)
* [快速上手](#快速上手)
  * [配置docker镜像](#配置docker镜像)
  * [安装python相关依赖](#安装python相关依赖)
  * [关键配置修改](#关键配置修改)
  * [运行示例](#运行示例)
    * [训练](#训练)
    * [推理](#推理)
* [高级](#高级)
  * [脚本参数](#脚本参数) 


## 概述

Qmix算法是一个多智能体强化学习算法，它通过集中学习一个联合动作-状态的总奖励值，用于指导合作环境下多智能体的分布式策略。本代码采用Qmix算法学习starCraft中的 2s_vs_1sc 地图的游戏策略。Ascend提供的Qmix是基于TensorFlow实现的版本。

参考论文：Rashid T, Samvelyan M, De Witt C S, et al. QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning[J]. arXiv preprint arXiv:1803.11485, 2018.

参考实现：https://github.com/oxwhirl/pymarl

## 要求

- 安装有昇腾AI处理器的硬件环境

- 由于PySC2 的starCraft仿真环境只能跑在X86机器上，需使用X86的Host 环境

## 默认配置

- 网络结构

  初始学习率为0.0005

  优化器：RMSPropOptimizer

  单卡batchsize：32

  buffersize：5000 trajectory

  折扣系数gamma：0.99

  target网络更新周期: 200

  GRU中的rnn_hidden_dim: 64

- 状态空间：

  global state shape：27

  obs状态shape： 17

  控制的agent数量：2

  episode 的长度limit：300

  更详细的状态信息请参考 pysc2 中2s_vs_1sc 地图的状态属性

- 动作空间：

  2s_vs_1sc 地图中每个agent的动作数量： 7

  其他map的动作空间请参考pysc2 的官方文档

- 训练数据的获取

  QMIX算法无需准备训练数据集，通过与StarCraft2仿真器实时交互获取训练数据，本脚本具备学习Starcraft2中的2s_vs_1sc游戏的策略。


## 快速上手

### 配置docker镜像

1. 下载并安装StarCraft2仿真环境（目前算法使用的版本是SC2.4.6.2.69232.zip）：

```
unzip -P iagreetotheeula SC2.4.6.2.69232.zip
```

以及安装对应的地图（SMAC_Maps.zip），需要注意的是SMAC的版本为v0.1-beta1,解压到 $SC2PATH/Maps 目录:

```
unzip SMAC_Maps.zip
mv SMAC_Maps StarCraftII/Maps/    # 需要将地图放在仿真器的特定目录
```

2. 启动docker， docker环境依赖的安装，请参考`docker/DockerFile`文件， 注意设置SC2PATH 环境变量为外部挂载进去的Starcraft2仿真环境

```
docker run -itd \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /var/log/npu/conf/slog:/var/log/npu/conf/slog \
    -v /var/log/npu/conf/profiling:/var/log/npu/conf/profiling \
    -v /var/log/npu/profiling/container/4:/var/log/npu/profiling/ \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools \
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
    -v `pwd`:/xt-src \
    -v /YOUR_PATH_IN_HOST/StarCraftII:/pymarl-sim/StarCraftII \    # 需要确认docker中该path 与 SC2PATH环境变量一致
    --network=host \
    -w /xt-src \
    --name=qmix_proj qmix_docker_image:1014 /bin/bash
```

3. 进入docker之后，需要确认 SC2PATH 环境变量指向 StarCraft的安装路径，并且安装SMAC的python 包为0.1.0b1版本，如

```
# env |grep SC2
SC2PATH=/YOUR/PATH/TO/StarCraftII
# pip list |grep SMAC
SMAC                   0.1.0b1
```

4. 安装依赖项

```bash
cd ModelZoo_QMIX_TF_NOAH
pip3 install -e .
```

### 安装python相关依赖

```
pip3 install numpy scipy pyyaml matplotlib imageio tensorboard-logger

pip3 install pygame jsonpickle==0.9.6  setuptools

# 自行下载安装sacred
# cd /install/sacred && python3 setup.py install

# 自行下载安装smac，版本为0.1.0-0b1
# install smac with 0.1.0-0b1 version


# x86 docker里面需要编译安装opencv， 算法验证的是opencv4.2.0 版本，自行下载安装
# opencv-4.2.0.zip 

pip3 install lz4 psutil tqdm pyzmq gym atari-py redis pyyaml fabric2

pip3 install zmq imageio matplotlib==3.0.3 Ipython tensorboardX

# for profiling 
pip3 install xlsxwriter xlrd tabulate openpyxl pandas

# tensorflow 1.15.0 等环境使用Ascend run包的默认环境
```

**完整的安装过程可参考 `docker/Dockerfile`文件**


### 关键配置修改

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


### 运行示例

#### 训练

单卡训练

- 设置SC2PATH环境变量

`SC2PATH=/YOUR/PATH/TO/StarCraftII`

- 启动训练脚本，示例如下。注意check SC2PATH 环境变量的位置指向starcraft的安装位置！

```
cd ModelZoo_QMIX_TF_NOAH
source env_ascend.sh
```

- 查看训练结果

```
tensorboard --logdir=~/xt_archive
```

正常情况下，需要4.5小时内可完成 205000 steps的训练，并收敛。其中`train_reward_avg`项收敛到6分以上。


#### 推理

测试的时候，需要修改配置文件`examples/ma_cases/qmix.yaml`，如下所示。同时，注意check SC2PATH 环境变量的位置指向starcraft2的安装位置！

```
node_config: [["127.0.0.1", "username", "passwd"],]      # set local node,
test_node_config: [["127.0.0.1", "username", "passwd"],]
test_model_path: /home/YOUR_PATH_TO_QMIX_ARCHIVE/models
benchmark:
  eval:
    gap:  25  # 256               # gap between models under test_model_path when call once evaluate
    evaluator_num: 1              # run eval with how much evaluator instance
    episodes_per_eval: 32         # run how much episodes within one evaluate
    max_step_per_episode: 1000
```

上述文件修改完成之后，执行测试代码，可以看到测试的数据信息

```
python3 xt/main.py -f examples/ma_cases/qmix.yaml -t evaluate -v debug
```


## 高级

### 脚本参数

配置文件为`examples/ma_cases/qmix.yaml`

```
alg_para:
  alg_name: QMixAlg               # The qmix algorithm defined within xingtian
  alg_config:                     # algorithm config
    batch_size: 32                # train batch size
    buffer_size: 5000             # train buffer  size of trajectory
    epsilon_anneal_time: 50000    # time_length of epsilon schedule
    epsilon_finish: 0.05          # epsilon minimum
    epsilon_start: 1.0            # epsilon maximum
    obs_agent_id: True            # use agent id within state
    obs_last_action: True         # use last action within state with onehot
    target_update_interval: 200   # interval to update target network

env_para:
  env_name: StarCraft2Xt      # The environment defined within xingtian
  env_info: {                 # follows are the starcraft simulator set
    "continuing_episode": False,
    "difficulty": "7",
    "game_version": null,
    "map_name": "2s_vs_1sc",   # map set
    "move_amount": 2,
    "obs_all_health": True,
    "obs_instead_of_state": False,
    "obs_last_action": False,
    "obs_own_health": True,
    "obs_pathing_grid": False,
    "obs_terrain_height": False,
    "obs_timestep_number": False,
    "reward_death_value": 10,
    "reward_defeat": 0,
    "reward_negative_scale": 0.5,
    "reward_only_positive": True,
    "reward_scale": True,
    "reward_scale_rate": 20,
    "reward_sparse": False,
    "reward_win": 200,
    "replay_dir": "",
    "replay_prefix": "",
    "state_last_action": True,
    "state_timestep_number": False,
    "step_mul": 8,
    "seed": null,
    "heuristic_ai": False,
    "heuristic_rest": False,
    "debug": False,
  }

agent_para:
  agent_name: StarCraftQMix     # The qmix agent defined within xingtian
  agent_num : 1                 # makeup multiagent as a whole
  agent_config: {
    'complete_step': 2050000    # explore step in total
    }

model_para:
  actor:
    model_name: QMixModel       # The qmix model defined within xingtian
    use_npu: False              # npu usage flag
    allow_mix_precision: True   # setup mix precision

    model_config:
      gamma: 0.99               # discount value for accumulative reward
      grad_norm_clip: 10        # clip value for grad norm
      hypernet_embed: 64        # size of each hypernet embed
      hypernet_layers: 2        # layers num of hupernet
      lr: 0.0005                # learning rate
      mixing_embed_dim: 32      # the dimensionality of mixing embed
      rnn_hidden_dim: 64        # the dimensionality of rnn hidden
      batch_size: 32            # train batch size for tensorboard model build
      use_double_q: True        # build model with double q algorithm

env_num: 1                      # explore environment number to parallel
node_config: [["127.0.0.1", "username", "passwd"],]        # set local node,
#test_node_config: [["127.0.0.1", "username", "passwd"],]  # train with eval, could see the eval rate

benchmark:
  id: xt_qmix
#  archive_root: ../xt_archive  # default: ~/xt_archive     # archive root set 
  eval:                           # eval config go with test_node_config activated
    gap: 256                      # train times call once evaluate
    evaluator_num: 1              # run eval with how much evaluator instance
    episodes_per_eval: 32         # run how much episodes within one evaluate
    max_step_per_episode: 1000    # max step within eval 
```


