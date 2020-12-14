# Bert Tnews for ACL

## 目录

* [目的](#目的)
* [环境安装依赖](#环境安装依赖)
* [数据集及模型准备](#数据集及模型准备)
* [模型转换](#模型转换)
* [demo编译](#demo编译)
* [离线推理](#离线推理)
* [备注](#备注)

## 目的

- 基于Bert网络进行文本的Tnews 新闻分类的离线推理

  

## 环境安装依赖

- 参考《驱动和开发环境安装指南》安装NPU训练环境




## 数据集及模型准备

FineTuning的代码路径：

[FineTuning代码参考](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/BertTnews_for_TensorFlow)

获取在线推理的数据集目录下的文件，拷贝到./datasets/bert_tnews_origin/目录下

```
cp ${FineTuning}/CLUEdataset/tnews/* ./datasets/bert_tnews_origin/
```

Pb模型：
基于Fine Tuning 获取bert_tnews checkpoint转pb 的pb模型,将pb模型放在model目录下



## 模型转换

离线模型转换

```
cd model
./bert_tnews_convert_om.sh --model=Bert_tnews --batchSize=1 --soc_version=Ascend310
```

该脚本用来做bert模型的atc转换，支持三个参数：

```
--batchSize     离线推理模型的batchSize，默认值为1
--soc_version   离线推理的soc_version，对于1910，这个参数设置为Ascend310，对于1980，这个参数设置为Ascend910，默认值为Ascend310
--model         模型名称，该模型名称不需要带后面的.pb 后缀，默认值为Bert_tnews
```



## demo编译

如果demo编译后在host侧运行，则执行以下脚本：

```
cd bert_infer
chmod +x build_host.sh
./build_host.sh
```

如果demo编译后在device侧运行，则执行以下脚本：

```
cd bert_infer
chmod +x build_device.sh
./build_device.sh
```


  可执行文件生成在../output目录

## 离线推理

端到端拉起离线推理：

```
./bert_tnews.sh --preprocess=1 --json=inference_syn_bert_b1.json
```

脚本中包含预处理，后处理和推理

#### 预处理脚本

```
./prerun_bert_tnews_infer_input.sh
```


​     export GLUE_DATA_DIR=$CURRENT_DIR/../datasets/bert_tnews_origin/  这个目录为环境上FineTuning的CLUE数据集的路径，根据实际修改

#### 推理脚本

```
./bert_infer inference_syn_bert_b1.json
```

​    json脚本中，涉及om_path_list，dir_path_list，result_path_list，json文件中给出的只是样例，建议根据环境上实际的路径替换修改
​    输出结果路径：./model1_dev_0_chn_0_results/bert/
​    精度数据：./output/perform_static_dev_0_chn_0.txt

#### 后处理脚本

```
./bert_tnews_accuracy_calc.sh
```

​    脚本中的这两个路径，可以根据实际情况修改
​    real_file="../datasets/bert_tnews_origin/dev.json"
​    label_file="../datasets/bert_tnews_origin/labels.json"
​    基线的精度值：55.9%

## 备注

执行中文解码时，需要关注本地的LANG设置应该为LANG=en_US.UTF-8,否则可能导致预处理解码失败
echo $LANG
执行以下命令，切换编码格式：
locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
说明：如果机器不支持该编码格式，直接使用转换好的bin文件，不要做预处理