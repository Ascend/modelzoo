一、训练流程：
    
单卡训练流程：

```
	1.安装环境
	2.修改run_1p.sh字段"data"为当前磁盘的数据集路径
	3.修改字段device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
	4.cd到run_1p.sh文件的目录，执行bash run_1p.sh单卡脚本， 进行单卡训练
```

	
多卡训练流程

```
	1.安装环境
	2.修改多P脚本中字段"data"为当前磁盘的数据集路径
	3.修改字段device_id_list（多卡训练所使用的device id列表），为训练配置device_id，比如4p,device_id_list=0,1,2,3；8P默认使用0，1，2，3，4，5，6，7卡不用配置
	4.cd到run_8p.sh文件的目录，执行bash run_8p.sh等多卡脚本， 进行多卡训练	
```



	
二、Docker容器训练：
    
1.导入镜像二进制包docker import ubuntuarmpytorch.tar REPOSITORY:TAG, 比如:

`docker import ubuntuarmpytorch.tar pytorch:b020`

2.执行docker_start.sh后带三个参数：步骤1生成的REPOSITORY:TAG；数据集路径；模型执行路径；比如：

`./docker_start.sh pytorch:b020 /train/imagenet /home/ResNet50`

3.执行步骤一训练流程（环境安装除外）
	
三、测试结果
    
训练日志路径：在训练脚本的同目录下result文件夹里，如：

/home/ResNet50/result/training_8p_job_20201121023601
	
自测数据（不同系统和机器性能可能存在差异，以下性能数据仅供参考）：

```
	1.A+K平台，EulerOS2.8，1卡测试，精度76.230，性能fps 1585 images/s
	2.A+K平台，EulerOS2.8，8卡测试，精度76.202，性能fps 10611 images/s
```

