# NAML-NPU 

## 环境变量
install_path设置为run包安装路径

## 数据集
准备数据集MINDsmall_train.zip, MINDsmall_dev.zip, MINDsmall_util.zip
执行路径下创建data文件夹
分别解压至data/train, data/valid, data/utils目录下

## 启动训练
执行脚本：

```shell
./run_naml_npu_1p.sh ${device_id}
```

其中，device_id为需要执行训练的device_id，若需要修改执行训练的epoch，请自行修改run_naml_npu_1p.sh中的参数

## 查看训练日志
tail -f查看log目录下存放的训练日志train_${device_id}.log

## 注意
保存h5时：
如果是arm环境，环境变量LD_LIBRARY_PATH需要增加/usr/include/hdf5/lib/这个路径，否则会报错：
ImportError(''save_model' requires h5py.')


