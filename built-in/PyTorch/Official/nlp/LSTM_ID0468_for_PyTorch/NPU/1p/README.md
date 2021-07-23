### 一、训练流程

单卡训练流程：

```
	1.安装torch环境、安装kaldi环境
	2.数据集timit_npu_profiling放到1p上层目录下
	3.修改字段device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
	4.cd到run_1p.sh文件的目录，执行bash run_1p.sh 2 单卡脚本， 进行单卡训练
```

多卡训练流程

```
	1.安装torch环境、安装kaldi环境
	2.数据集timit_npu_profiling放到8p上层目录下
	3.cd到run_8p.sh文件的目录，执行bash run_8p.sh 2 多卡脚本， 进行多卡训练
```

### 二、Docker容器训练

1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

```
    docker import ubuntuarmpytorch.tar pytorch:b020
```

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

```
    ./docker_start.sh pytorch:b020 /home/LSTM/NPU/data /home/LSTM
```

3.执行步骤一训练流程（环境安装除外）

三、测试结果

训练日志路径：在训练脚本的同目录下result文件夹里，如：

```
    /home/LSTM/result/training_8p_job_20201121023601
```

