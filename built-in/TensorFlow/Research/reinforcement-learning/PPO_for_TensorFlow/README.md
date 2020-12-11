## 一、配置docker镜像
```bash
1. 确认环境是否已导入版本镜像
[root@localhost package_1110]# docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu_arm          18.04               d663ce295a44        4 seconds ago       3.92GB

2. 启动docker镜像docker run -it --network=host --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/log/npu/conf/slog:/var/log/npu/conf/slog -v /var/log/npu/profiling/container/0:/var/log/npu/profiling/ -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /root/ModelZoo_PPO_TF_NOAH/:/home/ModelZoo_PPO_TF_NOAH/ --privileged=true --name=ppo_container_name ubuntu_arm:18.04 /bin/bash
说明：红色部分根据实际信息修改。

3. 重新开一个窗口，在Host侧将编译好的sumo拷入docker内，将container_name改为实际容器名称，如上面的ppo_container_name
docker cp sumo-master <container_name>:/home/ModelZoo_PPO_TF_NOAH

4. 容器中安装依赖项
apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
cd ./ModelZoo_PPO_TF_NOAH
从https://github.com/eclipse/sumo下载SUMO，并根据https://sumo.dlr.de/docs/Installing/Linux_Build.html本地编译SUMO。
cd ./ModelZoo_PPO_TF_NOAH/rl
pip install -e .
cd ./ModelZoo_PPO_TF_NOAH/flow
pip install -e .

5.修改./ModelZoo_DQN_TF_NOAH/env_ascend.sh前三行脚本如下
export SUMO_HOME="/home/ModelZoo_DQN_TF_NOAH/sumo-master"
export PATH="$SUMO_HOME/bin:$PATH"
```

### 二、设置环境变量
```bash
当版本为Atlas Data Center Solution V100R020C10时，请使用以下环境变量：
export install_path=/usr/local/Ascend/nnae/latest
# driver包依赖
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
#fwkacllib 包依赖
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export 
PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
export
PATH=${install_path}/fwkacllib/ccec_compiler/bin:{install_path}/fwkacllib/bin:$PATH
#tfplugin 包依赖
export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
# opp包依赖
export ASCEND_OPP_PATH=${install_path}/opp
```

## 三、开始训练
```bash
单卡训练
设置单卡训练环境（脚本位于ModelZoo_PPO_TF_NOAH），示例如下。
请确保ModelZoo_PPO_TF_NOAH文件夹放在/home目录下source env_ascend.sh

单卡训练指令（脚本位于ModelZoo_PPO_TF_NOAH/rl）
python3.7 xt/main.py -f examples/ma_cases/ma_figure8_ppo.yaml -t train

查看训练结果
tensorboard --logdir=./ppo_figure8
```

## 四、开始测试
```bash
打开配置文件rl/examples/ma_cases/ma_figure8_ppo.yaml修改如下内容，关闭node_config和model_path，打开test_node_config并制定test_model_path
#node_config: [["127.0.0.1", "username", "passwd"]]
test_node_config: [["127.0.0.1", "username", "passwd"]]
test_model_path: ./ppo_figure8/<model_path>/models

运行测试代码
python3.7 xt/main.py -f examples/ma_cases/ma_figure8_ppo.yaml -t evaluate
```
