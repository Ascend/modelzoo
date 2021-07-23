### 一、训练流程

单卡训练流程：

```
	1.安装环境
	2.执行如下命令：
        cd test
        bash train_full_1p.sh --data_path=数据集路径 --device_id=卡号          # 精度训练
        bash train_performance_1p.sh --data_path=数据集路径 --device_id=卡号   # 性能训练
```

多卡训练流程

```
	1.安装环境
	2.执行如下命令：
        cd test
        bash train_full_8p.sh --data_path=数据集路径         # 精度训练
        bash train_performance_8p.sh --data_path=数据集路径  # 性能训练
```