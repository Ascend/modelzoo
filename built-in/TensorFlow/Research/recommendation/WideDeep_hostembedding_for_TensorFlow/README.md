# MMOE-NPU 

## 数据集
将widedeep_hostembedding/criteo_tfrecord_a拷贝至执行脚本路径下
将widedeep_hostembedding/tfrecord_2020_1204_threshold_100拷贝至执行脚本路径下

## 启动单P训练
执行脚本：

```shell
./run_widedeep_1p.sh ${device_id}
```

其中，device_id为需要执行训练的device_id


## 启动8P训练
执行脚本：

```shell
./run_widedeep_8p.sh
```





