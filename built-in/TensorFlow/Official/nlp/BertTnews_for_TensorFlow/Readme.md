# Bert Tnews for Tensorflow

## 目录

* [目的](#目的)
* [环境设置](#环境设置)
* [FineTuning](#FineTuning)
* [predict](#predict)
* [checkpoint转pb](#checkpoint转pb)
* [pb在线推理](#pb在线推理)

## 目的

- 基于Bert 预训练好的模型，在1980 NPU上进行Tnews Fine Tuning

- TNEWS' 是指今日头条中文新闻（短文本）分类 Short Text Classificaiton for News

  

## 环境设置

- 参考《驱动和开发环境安装指南》安装1980 NPU训练环境

  安装CPU TensorFlow：tensorflow >= 1.11.0 

  启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

  [Ascend 910环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)



## FineTuning

执行Tnews的Fine Tuning

```
cd ./bert
./run_classifier_tnews.sh
```

Fine Tuning执行结束后会在当前目录生成 tnews_output目录，Fine Tuning的checkpoint就在这个目录下



## predict

基于checkpoint执行predict

```
cd ./bert
./run_classifier_tnews.sh predict
```



## checkpoint转pb

```
./run_bert_tnews_ckpt_to_pb.sh
```



## pb在线推理

```
cd onlineInference
python3 online_inference.py
```

当前这个PYTHON脚本支持以下参数：

```
--model_path     Fine Tuning的模型路径,可以根据实际值修改，默认值为：../tnews_output/Bert_tnews.pb
--data_dir       数据集的路径，默认值为：../../../CLUEdataset/tnews/
--vocab_file     idx2char的文件路径，默认值为：../../../CLUEdataset/tnews/vocab.txt
--output_dir     预处理文件的输出路径，默认值为：../../../CLUEdataset/
--pre_process    是否做预处理，默认值为True
--post_process   是否做后处理，默认值为True
--batchSize      在线推理的BatchSize，默认值为1
```

在线推理基于1980 NPU 性能和精度数据如下：

| 模型       | BatchSize | 数据类型 | 开发集(dev) | 性能（FPS） |
| ---------- | --------- | -------- | ----------- | ----------- |
| Bert_Tnews | 1         | FP16     | 56.19%      | 230.638     |
| Bert_Tnews | 8         | FP16     | 56.19%      | 446.778     |
