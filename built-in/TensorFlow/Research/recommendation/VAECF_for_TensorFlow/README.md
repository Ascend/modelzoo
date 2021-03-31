# VAECF-NPU 

## 环境变量
install_path设置为run包安装路径

## 数据集
将movielens_data/ml-20m/preprocessed目录拷贝到当前路径的data目录下

## 启动训练
执行脚本：

```shell
./run_vaecf_npu_1p.sh ${device_id}
```

其中，device_id为需要执行训练的device_id，若需要修改执行训练的epoch，请自行修改run_vaecf_npu_1p.sh中的参数

## 查看训练日志
tail -f查看log目录下存放的训练日志train_${device_id}.log



