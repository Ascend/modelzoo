## 一、配置docker镜像
```bash
1. 确认环境是否已导入版本镜像
[root@localhost package_1110]# docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu_arm          18.04               d663ce295a44        4 seconds ago       3.92GB

2. 启动docker镜像
docker run -it --network=host --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /var/log/npu/conf/slog:/var/log/npu/conf/slog -v /var/log/npu/profiling/container/0:/var/log/npu/profiling/ -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /root/ModelZoo_MUZERO_TF_NOAH/:/home/code -w /home --privileged=true --name=muzero_container_name ubuntu_arm:18.04 /bin/bash

3. 容器中安装opencv
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

建立软连接
find /usr/local/ -type f -name "cv2*.so"
得到查找结果 path
cd /usr/local/lib/python3.7/site-packages
ln -s path cv2.so

4.安装刑天
cd /home/code
pip3 install -e .
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
启动训练脚本：
cd /home/code/
source env.sh
xt_main -f examples/default_cases/muzero_pong_new.yaml

查看训练结果
cd ~
tensorboard --logdir=xt_archive
```
