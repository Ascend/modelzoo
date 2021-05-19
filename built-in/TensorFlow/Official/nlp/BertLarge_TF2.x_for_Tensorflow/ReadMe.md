## BertLarge执行指南

### 1.git路径

https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/BertLarge_TF2.x_for_Tensorflow

### 2.数据集路径

```sh
# 数据集路径
100.107.225.64
/turingDataset/01-NLP/BertLarge_TF2.x_for_Tensorflow/tfrecord
```

### 3.执行命令

```shell
cd $path/test

#1P 
./train_full_1p_4bs.sh --data_path=/home/tfrecord

#8P
./train_full_8p_32bs.sh --data_path=/home/tfrecord
```

### 4.开放参数

```shell
# npu_config
# 必选
--data_path		           source data of training

# 可选
--precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
--over_dump		           if or not over detection, default is False
--data_dump_flag		   data dump flag, default is 0
--data_dump_step		   data dump step, default is 10
--profiling		           if or not profiling for performance debug, default is False
--data_save_path           the path to save dump/profiling data, default is /home/data

#示例：
./train_full_1p_4bs.sh --precision_mode=allow_fp32_to_fp16 --data_path=/home/imagenet_TF --data_dump_flag=True --data_dump_step=1 --data_save_path=/home/data
```

### 5.结果校验

```shell
# 吞吐量
Final Performance images/sec ：
#总耗时(s)
E2E Training Duration sec :
#精度
Final Train Accuracy：
```



