## 用户使用手册



### 快速入门的例子

```shell
# 刑天项目的主目录
cd rl

# 训练任务
python3 main.py -f examples/default_cases/cartpole_ppo.yaml -t train

# 评估任务
python3 main.py -f examples/default_cases/cartpole_ppo.yaml -t evaluate
```



训练启动的时候会打印工作目录（workspace），模型和相关评估的结果会保存在该目录下

```zsh
workspace:
        /home/xwei/xt_archive/xt_cartpole+20200628101412
model will save under path:
        /home/xwei/xt_archive/xt_cartpole+20200628101412/models
```



### yaml配置文件的含义

```yaml
alg_para:                                               # 算法模块的参数
  alg_name: PPO                                         # 系统注册的算法名称，默认为类名

  alg_config:                                           
    process_num: 1                                      # 训练是否启用多进程
    only_save_best_model: True                          # 保存模型的策略（开发中）

env_para:                                               # 环境模块的参数
  env_name: GymEnv                                      # 系统注册的环境名称，默认为类名
  env_info: { 'name': CartPole-v0, 'vision': False}     # 仿真器的具体map/游戏名称

agent_para:                                             # Agent的参数
  agent_name: CartpolePpo                               # 系统注册的Agent名称，默认为类名
  agent_num : 1                                         # 生存在同一环境下的agent数量
  agent_config: {
    'max_steps': 200 ,                                  # 每个episode的交互步数
    'complete_step': 50000                              # 整个训练探索的最大步数
    }

model_para:                                             # 模型模块的参数
  actor:                                                # 算法默认包含一个名为actor的模型
    model_name:  ActorCriticPPO                         # 系统注册的模型名称，默认为类名
    state_dim: [4]                                      # 模型的输入空间维度
    action_dim: 2                                       # 模型的输出空间维度
    summary: False                                      # 是否打印模型结构信息

env_num: 1                                              # 每个节点下并行多实例explorer的数量
node_config: [["127.0.0.1", "username", "passwd"]]      # 各actor运行的节点信息

#test_node_config: [["127.0.0.1", "user", "passwd"]]    # 评估节点信息，可支持同时训练与评估
#test_model_path: ../xt_archive/model_data/cartpole_0   # 需进行评估的模型路径

# remote_env:                                           # 支持远端环境
#  conda: /home/jack/anaconda2/envs/xt_qmix             # 远端conda环境
#  env:                                                 # 支持设置远端环境变量
#    SC2PATH: /home/jack/xw-proj/marl_sim/StarCraftII
#    no_proxy: "192.168.1.*,127.0.0.1,10.*,.huawei.com,.huawei.net"

#benchmark:                                             # benchmark 信息
## ‘+’ 是ID中的连接符，如果字符中包含该字符， 系统将直接使用该ID，不会添加时间戳等信息。
#  id: xt_cartpole            # default: default_ENV_ALG ('+'.join([ID, START_time]))
#  archive_root: ./xt_archive # default: ~/xt_archive   # 评估信息归档的根目录，会自动分配
#  eval:
#    gap: 20                                            # 每训练多少次进行一次评估，并归档
#    episodes_per_eval: 2                               # 每次评估跑多少轮episode	 
#    evaluator_num: 1 	                                # 支持并行评估的实例数量设置
#    max_step_per_episode: 2000                         # 每次评估最大步数

```



默认使用 tensorboard 展示训练状态信息，并且将任务相关的records信息保存在`workspace` 目录下。

其中，bechmark目录下保存了该次训练任务的参数配置，train/eval reward等关键信息；

```zsh
/home/xwei/xt_archive/xt_cartpole+20200628101412/
|-- benchmark
|   |-- records.csv
|   `-- train_config.yaml
|-- events.out.tfevents.1593310452.SZX1000519853
|-- models
|   |-- actor_00000.h5
|   |-- actor_00001.h5
|   |-- actor_00002.h5
|   |-- actor_00003.h5
|   |-- actor_00004.h5
|   `-- actor_00005.h5
`-- train_records.json
```





> Note: 用户在客户中代码中调用刑天Api接口，并注册custom模块的功能正在开发中，敬请期待。

