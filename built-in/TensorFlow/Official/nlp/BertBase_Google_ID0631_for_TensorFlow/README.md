# BertBase_Google_for_TensorFlow
## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)


## 基本信息

- 发布者（Publisher）：huawei
- 应用领域（Application Domain）：Natural Language Processing
- 版本（Version）：1.1
- 修改时间（Modified） ：2021.07.17
- 大小（Size）：1.3G
- 框架（Framework）：TensorFlow 1.15.0
- 模型格式（Model Format）：ckpt
- 精度（Precision）：Mixed
- 处理器（Processor）：昇腾910
- 应用级别（Categories）：Official
- 描述（Description）：基于TensorFlow框架的BERT网络训练代码

## 概述

BERT模型的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，其中“双向”表示模型在处理某一个词时，它能同时利用前面的词和后面的词两部分信息。该工程提供了BERT BASE的训练代码

- 参考论文：

    [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

- 参考实现：

  https://github.com/google-research/bert

- 适配昇腾 AI 处理器的实现：
  
  
  https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/BertBase_Google_ID0631_for_TensorFlow


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

### 默认配置<a name="section91661242121611"></a>

- 训练超参

  - train_batch_size: 32
  - learning_rate: 1e-4
  - num_train_steps=100000
  - max_seq_length: 256
  - max_prediction_per_seq: 40
  - num_warmup_steps: 1000


### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，通过传入参数--precision_mode可以改变精度模式，相关代码示例：

```
flags.DEFINE_string("precision_mode", "allow_mix_precision", "The precision mode of training")

custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
```




## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 快速上手

### 数据集和预训练模型准备

  模型训练使用wikipedia数据集，参考源代码提供路径下载。

  预训练模型请参考源代码提供的链接下载，例如BERT-Base，Uncased。下载的文件夹中应该含有预处理模型，vocab.txt和bert_config.json。

  数据集和预训练模型下载完成后，放入模型目录下，在训练脚本中指定数据集和模型路径，可正常使用。

- 训练数据集预处理（以wikipedia为例，仅作为用户参考示例）：

  Sequence Length原则上用户可以自行定义

  本次训练设置为128，mask其中的40个tokens作为自编码恢复的目标。

  下游任务预处理以用户需要为准。

- 测试数据集预处理（以wikipedia为例，仅作为用户参考示例）：

  和训练数据集处理一致。

- 执行数据集预处理

  数据集以文本格式表示，每段之间以空行隔开，如wikipedia。 运行如下命令，将数据集转换为tfrecord格式。

  ```
  python3 create_pretraining_data.py \   
        --input_file=<path to your testdata> \   
        --output_file=<tfrecord dir>/some_output_data.tfrecord \   
        --vocab_file=<path to vocab.txt> \   
        --do_lower_case=True \   
        --max_seq_length=256 \   
        --max_predictions_per_seq=40 \   
        --masked_lm_prob=0.15 \   
        --random_seed=12345 \   
        --dupe_factor=5
  ```

### 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

启动单卡训练，data_path传入处理好的tf record格式数据集路径，type传入bertbase

- 单卡训练

  ```
  cd test
  bash train_full_8p.sh --type=bertbase --data_path=wikipedia
  ```
  
- 8卡训练

  ```
  cd test
  bash train_full_1p.sh --type=bertbase --data_path=wikipedia
  ```
  
  

## 高级参考

### 脚本和示例代码<a name="section08421615141513"></a>

```
└─Bert_Google_pretrain_for_TensorFlow
    ├─test
    |     ├─train_full_1p.sh
    |     ├─train_performance_1p.sh
    |     ├─train_performance_8p.sh
    |     ├─train_full_8p.sh
    |     └─env.sh
    ├─configs
    |   ├─bert_base_12layer_config.json
    |   └─bert_large_24layer_config.json
    ├─CONTRIBUTING.md
    ├─create_pretraining_data.py
    ├─extract_features.py
    ├─LICENSE
    ├─modeling.py
    ├─modeling_test.py
    ├─multilingual.md
    ├─optimization.py
    ├─optimization_test.py
    ├─README.md
    ├─run_classifier.py
    ├─run_classifier_with_tfhub.py
    ├─run_pretraining.py
    ├─run_squad.py
    ├─tokenization.py
    └─tokenization_test.py
```

### 脚本参数<a name="section6669162441511"></a>

```
    --precision_mode           precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                if or not over detection, default is False
    --data_dump_flag           data dump flag, default is False
    --data_dump_step           data dump step, default is 10
    --profiling                if or not profiling for performance debug, default is False
    --data_path                source data of training
    --type                     bertbase or bertlarge
    --num_warmup_steps         num_warmup_steps
    --save_checkpoints_steps   save_checkpoints_steps
    --train_batch_size         train_batch_size
    --learning_rate            learning_rate
    --num_train_steps          num_train_steps
    --bert_config_file         bert_config_file
    --output_dir               output_dir
    --max_predictions_per_seq  max_predictions_per_seq
    --max_seq_length           max_seq_length
```

### 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练和8卡训练。

2.  若不指定output_dir，训练日志及结果见test/output/${ASCEND_DEVICE_ID}的train_*.log



### 推理/验证过程<a name="section1465595372416"></a>

执行训练脚本默认执行验证过程，验证结果在test/output/${ASCEND_DEVICE_ID}的*_acc.log中打印

```
## 单p结果
AcutualFPS = 312
TrainMasked_lm_accuracy = 0.5498144
TrainNext_sentence_accuracy = 0.97875


## 8p结果
AcutualFPS = 1781
TrainMasked_lm_accuracy = 0.5466179
TrainNext_sentence_accuracy = 0.98

```
