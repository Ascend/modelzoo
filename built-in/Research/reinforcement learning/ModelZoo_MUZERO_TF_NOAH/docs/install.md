#### 刑天代码获取以及系统环境的建立

##### 获取代码

[黄区仓](https://gitlab.huawei.com/ee/train/rl)： `git clone ssh://git@gitlab.huawei.com:2222/ee/train/rl.git`



##### 系统库依赖

- redis
- Opencv


```shell
# ubuntu 18.04
sudo apt-get install python3-pip libopencv-dev redis-server -y
pip3 install opencv-python
```

##### Python 包依赖

```shell
cd rl
pip3 install -r requirements.txt

# 支持pip安装
pip3 install -e . 
```

> Note: 目前刑天平台稳定测试的 Tensorflow 版本为1.15.0版本，其他版本可能存在未知兼容性问题

