-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：NLP**

**版本（Version）：1.2**

**修改时间（Modified） ：2020.10.14**

**大小（Size）：1331.2M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的BERT-Base及下游任务代码**

<h2 id="概述.md">概述</h2>

   BERT是谷歌2018年推出的预训练语言模型结构，通过自监督训练实现对语义语境相关的编码，是目前众多NLP应用的基石。

-   参考论文：

    [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv 
    preprint arXiv:1810.04805.](https://arxiv.org/pdf/1810.04805.pdf)
        
-   参考实现：

    [https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

-   适配昇腾 AI 处理器的实现：
    
    
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Benchmark/nlp/Bert-base_for_TensorFlow



-   通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 网络结构
 
  学习率为1e-5，使用polynomial decay
 
  优化器：Adam

  优化器Weight decay为0.01

  优化器epsilon设置为1e-4

  单卡batchsize：128

  32卡batchsize：128*32

  总step数设置为500000

  Warmup step设置为10000

- 训练数据集预处理（以wikipedia为例，仅作为用户参考示例）：

  Sequence Length原则上用户可以自行定义

  以常见的设置128为例，mask其中的20个tokens作为自编码恢复的目标。
  
  下游任务预处理以用户需要为准。

- 测试数据集预处理（以wikipedia为例，仅作为用户参考示例）：

  和训练数据集处理一致。

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

开启混合精度相关代码示例。

```
    run_config = NPURunConfig(
            model_dir=self.config.model_dir,
            session_config=session_config,
            keep_checkpoint_max=5,
            save_checkpoints_steps=5000,
            enable_data_pre_proc=True,
            iterations_per_loop=iterations_per_loop,
            precision_mode='allow_mix_precision',
            hcom_parallel=True      
        ）
```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** _镜像列表_

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

-  单击“立即下载”，下载源码包。

-  数据集准备<a name="section361114841316"></a>

   数据集以文本格式表示，每段之间以空行隔开。源码包目录下“data/pretrain-toy/”给出了sample_text以及处理后的样例tfrecord数据集，如wikipedia。
   运行如下命令，将数据集转换为tfrecord格式。

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

- 启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

   [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

 - 单卡训练
    
    1. 在`scripts`路径下的`run_pretraining.py`中配置参数，确保 `--input_files_dir` 和 `--eval_files_dir` 配置为用户数据集具体路径，如下：
        
```
        --input_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \      #训练数据集路径
        --eval_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \       #验证数据集路径
```
  
      2. 单卡训练指令，在ModelZoo_BertBase_TF目录下，执行如下命令：
            
            bash scripts/run_pretraining.sh

   
- 8卡训练
    1. 在`scripts`路径下的`train_8p.sh`中配置参数，确保 `--input_files_dir` 和 `--eval_files_dir` 配置为用户数据集具体路径，如下：
        ```
         --input_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \      #训练数据集路径
         --eval_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \       #验证数据集路径  
         ```
    2. 8卡训练指令，在ModelZoo_BertBase_TF目录下，执行如下命令： 

        ```
        bash scripts/run_8p.sh
        ```

-  验证。

    1. 提供三个脚本，分别是文本分类任务，序列标注任务，阅读理解任务，并且提供了XNLI，LCQMC，CHNSENTI，NER，CMRC的数据处理方法示例，用户可根据自己的下游任务需要改写和处理数据。然后运行脚本，参考超参已经写入脚本供用户参考。
    
        执行命令：
        
        ```
        bash scripts/run_downstream_classifier.sh进行分类下游任务。
        
        bash scripts/run_downstream_ner.sh进行序列标注下游任务。
        
        bash scripts/run_downstream_reading.sh进行阅读理解下游任务。
        ```
    
    2. 执行命令前请先阅读相应bash脚本，补充相应文件路径。


<h2 id="高级参考.md">高级参考</h2>

    脚本和示例代码
    ├── configs  
    │    ├──BERT_base_64p_poc.json              //8*8p rank table配置文件
    │    ├──nezha_large_config.json               //NEZHA large模型配置文件
    │    ├──nezha_large_vocab.txt                 //NEZHA large中文词表
    ├── scripts
    │    ├──npu_set_env.sh                         //集群配置
    │    ├──run_downstream_classifier.sh           //运行下游任务分类器
    │    ├──run_downstream_ner.sh                  //运行下游任务序列标注
    │    ├──run_downstream_reading.sh              //运行下游任务阅读理解
    │    ├──run_pretraining.sh                     //单卡预训练脚本
    │    ├──run_8p.sh                              //8卡预训练入口脚本
    │    ├──train_8p.sh                            //8卡预训练脚本  
    ├── src/downstream
    │    ├──gpu_environment.py                     //原始gpu_environment设置
    │    ├──metrics_impl.py                       //适配NPU后的metrics_impl.py
    │    ├──modeling.py                           //NEZHA模型脚本
    │    ├──optimization.py                       //优化器脚本
    │    ├──reading_evaluate.py                   //阅读理解评价脚本
    │    ├──run_classifier.py                     //下游任务分类脚本
    │    ├──run_ner.py                           //下游任务序列标注脚本
    │    ├──run_reading.py                         //下游任务阅读理解脚本
    │    ├──tf_metrics.py                        //tf metrics脚本
    │    ├──tokenization.py                      //分词器脚本
    ├── src/pretrain
    │    ├──gpu_environment.py                     //原始gpu_environment设置
    │    ├──create_pretraining_data.py            //生成与训练数据脚本
    │    ├──modeling.py                           //NEZHA模型脚本
    │    ├──optimization.py                       //优化器脚本
    │    ├──extract_features.py                   //特征抽取脚本
    │    ├──fp16_utils.py                       //fp16 utils脚本
    │    ├──fused_layer_norm.py                     //layer norm融合脚本
    │    ├──run_pretraining.py                    //预训练启动脚本
    │    ├──tf_metrics.py                        //tf metrics脚本
    │    ├──tokenization.py                      //分词器脚本
    │    ├──utils.py                            //utils脚本├── CONTRIBUTING.md                             //CONTRIBUTING.md
    ├── src/downstream
    │    ├──gpu_environment.py                     //原始gpu_environment设置
    │    ├──metrics_impl.py                       //适配NPU后的metrics_impl.py
    │    ├──modeling.py                           //NEZHA模型脚本
    │    ├──optimization.py                       //优化器脚本
    │    ├──reading_evaluate.py                   //阅读理解评价脚本
    │    ├──run_classifier.py                     //下游任务分类脚本
    │    ├──run_ner.py                           //下游任务序列标注脚本
    │    ├──run_reading.py                         //下游任务阅读理解脚本
    │    ├──tf_metrics.py                        //tf metrics脚本
    │    ├──tokenization.py                      //分词器脚本
    ├── src/pretrain
    │    ├──gpu_environment.py                     //原始gpu_environment设置
    │    ├──create_pretraining_data.py            //生成与训练数据脚本
    │    ├──modeling.py                           //NEZHA模型脚本
    │    ├──optimization.py                       //优化器脚本
    │    ├──extract_features.py                   //特征抽取脚本
    │    ├──fp16_utils.py                       //fp16 utils脚本
    │    ├──fused_layer_norm.py                     //layer norm融合脚本
    │    ├──run_pretraining.py                    //预训练启动脚本
    │    ├──tf_metrics.py                        //tf metrics脚本
    │    ├──tokenization.py                      //分词器脚本
    │    ├──utils.py                            //utils脚本
    ├── CONTRIBUTING.md                             //CONTRIBUTING.md
    ├── LICENCE                                   //LICENCE
    ├── NOTICE                                   //NOTICE├── README.md                                 //说明文档
    

## 脚本参数<a name="section6669162441511"></a>
    
     
```
         --train_batch_size=128 \           #每个NPU训练的batch size，默认：128
         --learning_rate=1e-4 \             #学习率，默认：1e-4
         --num_warmup_steps=10000 \         # 初始warmup训练epoch数，默认：10000
         --num_train_steps=500000 \         #训练次数，单P 默认：500000
         --input_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \      #训练数据集路径
         --eval_files_dir=/autotest/CI_daily/ModelZoo_BertBase_TF/data/wikipedia_128 \       #验证数据集路径  
         --iterations_per_loop=100 \        #NPU运行时，device端下沉次数，默认：1000 
    
```


## 训练过程<a name="section1589455252218"></a>

通过“快速上手”中的训练指令启动训练。

```
I0521 19:45:05.731803 281473752813584 basic_session_run_hooks.py:692] global_step/sec: 2.451
I0521 19:45:05.732023 281473228546064 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.7687549, next_sentence_loss = 0.005564222, total_loss = 0.7743191 (81.600 sec)
I0521 19:45:05.732058 281473117769744 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.74314255, next_sentence_loss = 0.023222845, total_loss = 0.7663654 (81.600 sec)
2020-05-21 19:45:05.732132: I tf_adapter/kernels/geop_npu.cc:526] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp15_0[ 2409us]
I0521 19:45:05.732016 281473584246800 basic_session_run_hooks.py:692] global_step/sec: 2.451
I0521 19:45:05.732048 281472971046928 basic_session_run_hooks.py:692] global_step/sec: 2.451
loss_scale: loss_scale:[65536.0] 
2020-05-21 19:45:05.732378: I tf_adapter/kernels/geop_npu.cc:526] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp15_0[ 2445us]
loss_scale:[65536.0] 
I0521 19:45:05.732480 281473752813584 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.94164073, next_sentence_loss = 0.023505606, total_loss = 0.96514636 (81.600 sec)
I0521 19:45:05.732715 281473584246800 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.738043, next_sentence_loss = 0.03810045, total_loss = 0.77614343 (81.599 sec)
I0521 19:45:05.732658 281473385623568 basic_session_run_hooks.py:692] global_step/sec: 2.451
I0521 19:45:05.732574 281473416220688 basic_session_run_hooks.py:692] global_step/sec: 2.45098
I0521 19:45:05.732777 281472971046928 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.7797201, next_sentence_loss = 0.05669275, total_loss = 0.8364129 (81.600 sec)loss_scale: [65536.0]
loss_scale:[65536.0] 
I0521 19:45:05.733291 281473385623568 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.8004036, next_sentence_loss = 0.12787658, total_loss = 0.9282802 (81.600 sec)[65536.0]

```

调优过程

通过“快速上手”中的调优说明(即验证章节)，对自己的下游任务进行调优和预测。

## 推理/验证过程<a name="section1465595372416"></a>

见下游任务Finetune（“快速上手”中的验证章节）。

