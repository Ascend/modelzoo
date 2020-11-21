## 一、 依赖安装

1) 安装 StarCraft2 仿真环境， 可参考https://github.com/Blizzard/s2client-proto#downloads ， 目前算法使用的版本是`SC2.4.6.2.69232.zip`

```
# 下载参考命令
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
unzip -P iagreetotheeula SC2.4.6.2.69232.zip

# 安装对应的map，需要注意的是SMAC的版本为v0.1-beta1
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip

mv SMAC_Maps StarCraftII/Maps/

# 设置SC2PATH的环境变量
export SC2PATH=YOUR_PATH_TO/StarCraftII
```

2）安装python相关依赖

```
pip3 install numpy scipy pyyaml matplotlib imageio tensorboard-logger

pip3 install pygame jsonpickle==0.9.6  setuptools

git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#install smac with 0.1.0-0b1 version
https://github.com/oxwhirl/smac/tree/v0.1-beta1

# x86 docker里面需要编译安装opencv， 算法验证的是opencv4.2.0 版本
# opencv-4.2.0.zip from https://github.com/opencv/opencv/archive/4.2.0.zip

pip3 install lz4 psutil tqdm pyzmq gym atari-py redis pyyaml fabric2

pip3 install zmq imageio matplotlib==3.0.3 Ipython tensorboardX

# for profiling 
pip3 install xlsxwriter xlrd tabulate openpyxl pandas

# tensorflow 1.15.0 等环境使用Ascend run包的默认环境
```

**完整的安装过程可参考 `docker/Dockerfile`文件**



3） 安装qmix 代码

```bash
cd ModelZoo_QMIX_TF_NOAH
pip3 install -e .
```



### 二、 运行代码

```bash
source env_ascend.sh
```

正常情况下，需要4.5小时内可完成 205000 steps的训练，并收敛。


### 三、 收敛效果

查看tensorboard

```
tensorboard --logdir=~/xt_archive
```

其中`train_reward_avg`项收敛到6分以上。

