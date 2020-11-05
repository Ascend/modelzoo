## 一、 依赖安装

```bash
apt-get install sumo sumo-tools sumo-doc
cd ModelZoo_DDPG_TF_NOAH/rl
pip install -e .
```



### 二、 运行代码

```bash
source env_ascend.sh
```



### 三、 收敛效果

查看tensorboard

```
tensorboard --logdir=~/xt_archive
```

其中train_reward_avg项收敛到30。
