## 一、环境

1. tensorflow-1.15
2. Ascend910
3. [subword-NMT](https://github.com/rsennrich/subword-nmt)

## 二、模型介绍

本仓库是经典的RNNsearch机器翻译模型，按照[leaderboard](https://paperswithcode.com/sota/machine-translation-on-iwslt2015-german?p=pervasive-attention-2d-convolutional-neural-1)中的名称，也可将其简称为BiGRU。模型出自[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)。

鉴于原文没有说明在这个测试集上的表现，且原作者没有公开官方代码，本仓库主要借鉴：[清华大学复现的RNNsearch仓库](https://github.com/THUNLP-MT/THUMT/tree/tensorflow)。由于NPU暂时不支持dynamic shape特性和部分资源算子，此仓库已经改写为静态shape版本。

## 三、训练与测试

### 执行预处理

主要依赖subword-NMT进行文本的预处理。

1. 下载En-De数据集

   下载WMT17上的预处理平行语料：[链接](http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/)

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
   python subword-nmt/learn_joint_bpe_and_vocab.py --input corpus.tc.de corpus.tc.en -s 32000 -o bpe32k --write-vocabulary vocab.de vocab.en
   ```

4. 对训练集、验证集进行BPE分词

   ```shell
   python subword-nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k corpus.tc.de corpus.tc.32k.de
   python subword-nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k corpus.tc.en corpus.tc.32k.en
   python subword-nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < newstest2015.tc.de > newstest2015.tc.32k.de
   ```

5. 对训练集进行shuffle预处理（可选）

   ```shell
   python thumt/scripts/shuffle_corpus.py --corpus corpus.tc.32k.de corpus.tc.32k.en --suffix shuf
   ```

### 执行训练

```shell
. train_beta.sh
```

训练保存的模型位于/bi-gru/train/。

### 执行测试

```
. validate_beta.sh
```

如果同时加载training model和inference model，会导致单卡NPU显存溢出，所以训练过程需要训练人员手动在另一块NPU卡上执行validate或者指定时间间隔自动执行validate，来得到最新模型的BLEU值。

## 四、下载

提供训练好的checkpoint(model.ckpt-263002，BLEU值为26.7624)，请放到bi-gru/train/路径下：

百度云：[链接](https://pan.baidu.com/s/1-z7CX7F_FujKHrSdzfvs6g )
提取码：zal9 


## 五、性能

本仓库建模采用静态shape，这意味着每个GRU模块都将输入文本padding到固定的最长序列长度，再执行计算。这潜在地造成训练速度的降低。

平均耗时如下:

- GPU：0.454 sec/step，和动态shape版本的最长序列单步耗时0.46 sec/step相当
- NPU：2.355 sec/step 2020.12.15（2020.12.27降低为1.355 sec/step）

训练需要运行30w step，这代表在NPU上需要训练约合9天时间（2020.12.27降低为约合5天）。

## 六、精度

测试数据为iwslt2015 German-English dev set，这里讨论的均是iwslt 2015 German-English dev set的BLEU精度，如下表：

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

此外，由于28.53这个性能没有具体的出处和细节，为了确定dynamic-GPU-27.54作为模板是否合适，我们又对比了其他BiGRU（RNNsearch）复现版本在iwslt 2015 German-English的性能，全网只有2个results：

- book release：[beam-size=12, BLEU-27.25](https://books.google.com/books?id=KIOrDwAAQBAJ&pg=PA66&lpg=PA66&dq=newstest2015+rnnsearch&source=bl&ots=vzXUqjeYW_&sig=ACfU3U04ka_Rq-RCUeh5Ghd3BmIvCOhjgg&hl=zh-CN&sa=X&ved=2ahUKEwiZuISf7PLtAhVDwFkKHek3D4kQ6AEwCHoECAcQAg#v=onepage&q=newstest2015%20rnnsearch&f=false)
- google release：[beam-size=12, BLEU-20.5](https://google.github.io/seq2seq/results/)





## MSAME推理

安装号MSAME工具后，运行：

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

