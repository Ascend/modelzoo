# NEZHA-Large for TensorFlow

## 概述

NEZHA是华为诺亚方舟实验室推出的预训练语言模型结构，和BERT类似，并有诸多改进，目前在中英文自然语言理解榜单上均超越BERT，是目前最先进的中文预训练语言模型和最先进的英文预训练单模型。

参考论文：Wei, J., Ren, X., Li, X., Huang, W., Liao, Y., Wang, Y., ... & Liu, Q. (2019). NEZHA: Neural contextualized representation for chinese language understanding. arXiv preprint arXiv:1909.00204.

参考实现：https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow

## 默认配置
网络结构

学习率为1e-4，使用polynomial decay

优化器：Lamb

优化器Weight decay为0.01

优化器epsilon设置为1e-4

单卡batchsize：64

80卡batchsize：64*80

总step数设置为1000000

Warmup step设置为10000

训练数据集预处理（以wikipedia为例，仅作为用户参考示例）：

Sequence Length原则上用户可以自行定义
以常见的设置128为例，mask其中的20个tokens作为自编码恢复的目标。

下游任务预处理以用户需要为准。

测试数据集预处理（以wikipedia为例，仅作为用户参考示例）：

和训练数据集处理一致。

## 快速上手

1、数据集准备，该模型兼容tensorflow官网上的数据集。
数据集以文本格式表示，每段之间以空行隔开。源码包目录下“data/pretrain-toy/”中给出了sample_text以及处理后的样例tfrecord数据集，如wikipedia。
运行如下命令，将数据集转换为tfrecord格式。
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
原则上NEZHA只能用集群进行训练，以NEZHA-Large为例，至少需要以8*8p的集群规模训练若干天。具体训练时间以您的数据集大小为准。配置多级多卡分布式训练，需要您修改configs目录下NEZHA_large_64p_poc.json配置文件，将对应IP修改为您的集群对应的IP。

## 环境配置

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

多P训练时，需要依次拉起所有训练进程，因此需要在每个训练进程启动前，需要分别设置DEVICE_ID和RANK_ID，例如：
export DEVICE_ID=1
export RANK_ID=1

## 开始训练

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

## 下游任务Finetune
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


## 训练过程

通过“快速上手”中的训练指令启动训练。

## 验证/推理过程

见下游任务Finetune。