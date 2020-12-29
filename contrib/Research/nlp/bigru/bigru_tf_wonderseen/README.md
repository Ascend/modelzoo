## 一、环境

1. tensorflow-1.15
2. Ascend910
3. [subword-NMT](https://github.com/rsennrich/subword-nmt)

## 二、模型介绍

RNNsearch是经典的机器翻译模型，模型出自[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)。按照[iwslt 2015 leaderboard](https://paperswithcode.com/sota/machine-translation-on-iwslt2015-german?p=pervasive-attention-2d-convolutional-neural-1)中的名称，也可将其简称为BiGRU。

本仓库负责将BiGRU从tensorflow迁移至Ascend910，对标的精度是iwslt 2015 German-English test set BLEU值。鉴于原论文作者没有公开官方代码，文中也没有纰漏在该数据集上的细节，本仓库主要借鉴了科研单位复现的版本：[清华大学复现的RNNsearch仓库](https://github.com/THUNLP-MT/THUMT/tree/tensorflow)。NPU算子暂时不支持dynamic shape特性，此仓库是已经改写为静态shape的版本。

## 三、文件结构

```shell
bigru
└─ 
  ├─README.md
  ├─train.sh	训练脚本
  ├─validate.sh	测试脚本
  ├─thumt 		存放bigru所有代码的文件夹
  	└─...
  ├─train 		存放训练模型的默认文件夹
  	├─eval 		存放模型测试结果的默认文件夹
  	├─checkpoint
  	├─model.ckpt-*.data-00000-of-00001
  	├─model.ckpt-*.index
  	├─model.ckpt-*.meta
  	└─...
  ├─msame 		负责msame离线推理的文件夹
  	├─ckpt2pb.sh				将ckpt模型冻结成pb模型
  	├─pb2om.sh					将pb模型转为om模型
  	├─transform_ckpt_to_om.sh 	整合ckpt2pb.sh和pb2om.sh脚本
  	├─process_phrase_1.sh 		输入bin文件给om模型，保存推理结果
  	├─process_phrase_2.sh 		验证推理精度
  	├─run_msame_testing.sh		以上离线推理所有步骤的一脚启动脚本
  	├─msame						编译好的msame推理工具
  	└─...
```

## 四、训练与测试

### （1）执行预处理

文本的预处理主要依赖subword-NMT，按照以下流程：

1. 下载De-En翻译数据集。

   预处理平行语料和测试集：[链接](http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/)

   - 训练集

   ```shell
   gunzip corpus.tc.de.gz corpus.tc.en.gz
   ```

   得到 corpus.tc.* 作为训练集，共5,852,458对平行语句。

   - 验证集

   ```shell
    tar xvfz dev.tgz
   ```

   解压可以得到iwslt2015验证集 newstest2015.tc.* 

2. 下载subword-nmt

   ```shell
   git clone https://github.com/rsennrich/subword-nmt.git
   ```

3. 生成BPE operations

   ```shell
   python subword-nmt/learn_joint_bpe_and_vocab.py \
   --input corpus.tc.de corpus.tc.en \
   -s 32000 \
   -o bpe32k \
   --write-vocabulary vocab.de vocab.en
   ```

4. 对训练集、测试集进行BPE分词

   ```shell
   python subword-nmt/apply_bpe.py \
   --vocabulary vocab.de \
   --vocabulary-threshold 50 \
   -c bpe32k < corpus.tc.de > corpus.tc.32k.de
   
   python subword-nmt/apply_bpe.py \
   --vocabulary vocab.en \
   --vocabulary-threshold 50 \
   -c bpe32k < corpus.tc.en > corpus.tc.32k.en
   
   python subword-nmt/apply_bpe.py \
   --vocabulary vocab.de \
   --vocabulary-threshold 50 \
   -c bpe32k < newstest2015.tc.de > newstest2015.tc.32k.de
   ```

5. 对训练集进行shuffle预处理（可选）

   ```shell
   python thumt/scripts/shuffle_corpus.py --corpus corpus.tc.32k.de corpus.tc.32k.en --suffix shuf
   ```

6. 经过以上预处理后的语料和词表，请统一转移至/thumt/data/。

### （2）执行训练

```shell
. train_beta.sh
```

训练保存的模型位于train/。

### （3）执行测试

```
. validate_beta.sh
```

如果同时加载training model和inference model，会导致单卡NPU显存溢出，所以训练过程需要训练人员手动在另一块NPU卡上执行validate或者指定时间间隔自动执行validate，来得到最新模型的BLEU值。

## 五、下载

提供训练好的checkpoint (model.ckpt-263002，BLEU值为26.7624)，请放到 train/ 路径下：

百度云：[链接](https://pan.baidu.com/s/1-z7CX7F_FujKHrSdzfvs6g )

提取码：zal9 


## 六、性能

本仓库建模采用静态shape，这意味着每个GRU模块都将输入文本padding到固定的最长序列长度，再执行计算。这潜在地造成训练速度的降低。

平均耗时如下:

- GPU：0.454 sec/step，和动态shape版本的最长序列单步耗时0.46 sec/step相当
- NPU：2.355 sec/step 2020.12.15（2020.12.27降低为1.355 sec/step）

训练需要运行30w step，这代表在NPU上需要训练约合9天时间（2020.12.27降低为约合5天）。

## 七、精度

测试数据为iwslt2015 German-English test set，这里讨论的均是iwslt 2015 German-English test set的BLEU精度，如下表：

|             | original-GPU | dynamic-GPU | static-GPU | static-NPU |
| ----------- | ------------ | ----------- | ---------- | ---------- |
| beam_size=1 | -            | 26.76       | 27.20      | 26.76      |
| beam_size=4 | -            | 27.54       | -          | -          |
| unknown     | 28.53        | -           | -          | -          |

表格第一行均按照 <模型类型>-<训练环境> 格式：

- original-GPU：原论文作者实现的版本（代码和细节不公开）
- dynamic-GPU：清华实现的开源版本（默认为动态shape）
- static-GPU：根据dynamic-GPU重写的静态shape、且保证能在NPU上打通的版本
- static-NPU：根据dynamic-GPU重写的静态shape、且能在NPU上稳定训练的版本

另外，iwslt2015 German-English test set始终没有在原始论文中出现过，其性能BLEU-28.53所使用的语料、词表大小、模型大小等均没有任何说明。
为了确定dynamic-GPU-27.54 作为模板是否合适，可参考对比其他BiGRU（RNNsearch）复现版本在iwslt 2015 German-English的精度结果：

- book release：[beam-size=12, BLEU-27.25](https://books.google.com/books?id=KIOrDwAAQBAJ&pg=PA66&lpg=PA66&dq=newstest2015+rnnsearch&source=bl&ots=vzXUqjeYW_&sig=ACfU3U04ka_Rq-RCUeh5Ghd3BmIvCOhjgg&hl=zh-CN&sa=X&ved=2ahUKEwiZuISf7PLtAhVDwFkKHek3D4kQ6AEwCHoECAcQAg#v=onepage&q=newstest2015%20rnnsearch&f=false)
- google release：[beam-size=12, BLEU-20.5](https://google.github.io/seq2seq/results/)

## 八、MSAME离线推理

按照wiki编译好MSAME工具后，运行：

```shell
cd msame
. run_msame_testing.sh
```

预期输出：

```pascal
############### MSAME BLEU testing #############
Golden Translation at: msame/golden/references/
BiGRU Translation at: msame/output_offline/
BLEU = 26.759709157237165
```