# NEZHA-Large for TensorFlow

## 目录

* [概述](#概述)
* [要求](#要求)
* [默认配置](#默认配置)
* [快速上手](#快速上手)
  * [准备数据集](#准备数据集)
  * [Docker容器场景](#Docker容器场景)
  * [关键配置修改](#关键配置修改)
  * [运行示例](#运行示例)
    * [训练](#训练)
    * [推理](#推理)    


## 概述

NEZHA是华为诺亚方舟实验室推出的预训练语言模型结构，和BERT类似，并有诸多改进，目前在中英文自然语言理解榜单上均超越BERT，是目前最先进的中文预训练语言模型和最先进的英文预训练单模型。

参考论文：Wei, J., Ren, X., Li, X., Huang, W., Liao, Y., Wang, Y., ... & Liu, Q. (2019). NEZHA: Neural contextualized representation for chinese language understanding. arXiv preprint arXiv:1909.00204.

参考实现：https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow

## 要求

- 安装有昇腾AI处理器的硬件环境
- 下载并预处理wikipedia数据集以进行培训和评估。


## 默认配置

- 网络结构

  学习率为1e-4，使用polynomial decay

  优化器：Lamb

  优化器Weight decay为0.01

  优化器epsilon设置为1e-4

  单卡batchsize：64

  80卡batchsize：64*80

  总step数设置为1000000

  Warmup step设置为10000

- 训练数据集预处理（以wikipedia为例，仅作为用户参考示例）：

  Sequence Length原则上用户可以自行定义

  以常见的设置128为例，mask其中的20个tokens作为自编码恢复的目标。

  下游任务预处理以用户需要为准。

- 测试数据集预处理（以wikipedia为例，仅作为用户参考示例）：

  和训练数据集处理一致。

## 快速上手

### 准备数据集

- 请用户自行下载数据集，如wikipedia。数据集以文本格式表示，每段之间以空行隔开。

- 运行如下命令，将数据集转换为tfrecord格式。

```
python utils/create_pretraining_data.py \   
  --input_file=./your/path/some_input_data.txt \   
  --output_file=/data/some_output_data.tfrecord \   
  --vocab_file=./your/path/vocab.txt \   
  --do_lower_case=True \   
  --max_seq_length=128 \   
  --max_predictions_per_seq=20 \   
  --masked_lm_prob=0.15 \   
  --random_seed=12345 \   
  --dupe_factor=5
```

### Docker容器场景

- 编译镜像
```bash
docker build -t ascend-nezha .
```

- 启动容器实例
```bash
bash scripts/docker_start.sh
```

参数说明:

```
#!/usr/bin/env bash
docker_image=$1 \   #接受第一个参数作为docker_image
data_dir=$2 \       #接受第二个参数作为训练数据集路径
model_dir=$3 \      #接受第三个参数作为模型执行路径
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #docker使用卡数，当前使用0~7卡
 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #训练数据集路径
        -v ${model_dir}:${model_dir} \  #模型执行路径
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \     #docker_image为镜像名称
        /bin/bash
```

执行docker_start.sh后带三个参数：
  - 生成的docker_image
  - 数据集路径
  - 模型执行路径
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```



### 关键配置修改

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

原则上NEZHA只能用集群进行训练，以NEZHA-Large为例，至少需要以8*8p的集群规模训练若干天。具体训练时间以您的数据集大小为准。配置多级多卡分布式训练，需要您修改configs目录下NEZHA_large_64p_poc.json配置文件，将对应IP修改为您的集群对应的IP。

多P训练时，需要依次拉起所有训练进程，因此需要在每个训练进程启动前，需要分别设置DEVICE_ID和RANK_ID，例如：
export DEVICE_ID=1
export RANK_ID=1

### 运行示例

#### 训练

在`scripts`路径下的`train_8p.sh`中配置参数，确保 `--input_files_dir` 和 `--eval_files_dir` 配置为用户数据集具体路径。

参数说明：

```
python3.7 ${dname}/src/pretrain/run_pretraining.py \
 --bert_config_file=${dname}/configs/nezha_large_config.json \
 --max_seq_length=128 \
 --max_predictions_per_seq=20 \
 --train_batch_size=64 \
 --learning_rate=1e-4 \
 --num_warmup_steps=10000 \
 --num_train_steps=1000000 \
 --optimizer_type=lamb \
 --manual_fp16=True \
 --use_fp16_cls=True \
 --input_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \         #训练数据集路径
 --eval_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \          #验证数据集路径
 --npu_bert_debug=False \
 --npu_bert_use_tdt=True \
 --do_train=True \
 --num_accumulation_steps=1 \
 --npu_bert_job_start_file= \
 --iterations_per_loop=100 \
 --save_checkpoints_steps=10000 \
 --npu_bert_clip_by_global_norm=False \
 --distributed=True \
 --npu_bert_loss_scale=0 \
 --output_dir=./output > ${currentDir}/result/8p/train_${device_id}.log 2>&1
 ```

1.单卡训练

```
cd scripts

./run_pretraining.sh
```

2.8卡训练

```
cd scripts

./run_8p.sh
```


#### 推理
提供三个脚本，分别是文本分类任务，序列标注任务，阅读理解任务，并且提供了XNLI，LCQMC，CHNSENTI，NER，CMRC的数据处理方法示例。用户可根据自己的下游任务需要改写和处理数据。然后运行脚本，参考超参已经写入脚本供用户参考。

执行命令：

```
bash scripts/run_downstream_classifier.sh
进行分类下游任务。

bash scripts/run_downstream_ner.sh
进行序列标注下游任务。

bash scripts/run_downstream_reading.sh
进行阅读理解下游任务。
```

执行命令前请先阅读相应bash脚本，补充相应文件路径。



