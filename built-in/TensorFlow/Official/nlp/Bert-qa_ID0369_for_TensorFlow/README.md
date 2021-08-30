-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.8.15**

**大小（Size）：105KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的bert-qa无监督语言表示学习算法训练代码**

<h2 id="概述.md">概述</h2>

-    BERT是一种预训练语言表示的方法，这意味着我们在大型文本语料库（例如Wikipedia）上训练通用的“语言理解”模型，然后将该模型用于我们关心的下游NLP任务（例如问题回答）。BERT优于以前的方法，因为它是第一个用于预训练NLP的*无监督*，*深度双向*系统。

    -   参考论文：

        https://arxiv.org/abs/1810.04805

    -   参考实现：
        
        ```
        https://github.com/chiayewken/bert-qa
        ```
    
-   适配昇腾 AI 处理器的实现： 
    
        ```
        https://github.com/Ascend/modelzoo/blob/master/built-in/TensorFlow/Official/nlp/Bert-qa_ID0369_for_TensorFlow
        ```


    -   通过Git获取对应commit_id的代码方法如下：
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

## 默认配置 <a name="section91661242121611"></a>
-   网络结构
    12-layer, 768-hidden, 12-heads, 110M parameters

-   训练超参（单卡）：
    -   max_seq_length: 128
    -   max_predictions_per_seq: 20
    -   train_batch_size: 32
    -   learning_rate: 5e-5
    -   num_train_steps: 100000
    -   num_warmup_steps: 10000
    -   iterations_per_loop: 1000
    -   num_tpu_cores: 8


## 支持特性 <a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练 <a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度 <a name="section20779114113713"></a>
相关代码示例。



```
  #for NPU
  session_config = tf.ConfigProto()
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  #for NPU
  run_config = tf.contrib.tpu.RunConfig(
      session_config = session_config,
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
```

<h2 id="训练环境准备.md">训练环境准备</h2>

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1. 模型预训练使用自带数据集，vocab.txt需用户自行下载。

2. 这里是列表文本数据集训练前需要做预处理操作。

   ```
   cd ./Bert-qa_ID0369_For_Tensorflow
   bash create_pretraining_data.sh
   (脚本中需自行配置：--vocab_file)
   ```
   
3. 这里是列表文本数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练 <a name="section715881518135"></a>
- 下载训练脚本。

- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于./Bert-qa_ID0369_For_Tensorflow/test/train_performance_1p.sh），示例如下。请确保下面例子中的“input_file和bert_config_file”修改为用户数据集的路径。
            
        
        ```
        `nohup python3 run_pretraining.py \
          --input_file=./tf_examples.tfrecord \
          --output_dir=./pretraining_output \
          --do_train=True \
          --do_eval=True \
          --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
          --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
          --train_batch_size=32 \
          --max_seq_length=128 \
          --max_predictions_per_seq=20 \
          --num_train_steps=20 \
          --num_warmup_steps=10 \
          --learning_rate=2e-5`
        ```
        
        
        
        2.2 单卡训练指令（脚本位于./Bert-qa_ID0369_For_Tensorflow/test/train_performance_1p.sh） 

```
        `bash train_performance_1p.sh`
```



<h2 id="高级参考.md">高级参考 </h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt						 //依赖
    ├──test										 
    │    ├──train_performance_1p.sh				 //单卡训练脚本
    │    ├──env.sh								 //环境变量
    ├──albert_config                     	     //网络配置
    ├──create_pretraining_data.sh           	 //预处理执行脚本
    ├──create_pretraining_data.py          	     //预处理脚本
    ├──run_pretraining.py              		     //预训练脚本


## 脚本参数 <a name="section6669162441511"></a>

```
    --data_dir                        train data dir, default : path/to/data
    --num_classes                     number of classes for dataset. default : 1000
    --batch_size                      mini-batch size ,default: 128 
    --lr                              initial learning rate,default: 0.06
    --max_epochs                      total number of epochs to train the model:default: 150
    --warmup_epochs                   warmup epoch(when batchsize is large), default: 5
    --weight_decay                    weight decay factor for regularization loss ,default: 1e-4
    --momentum                        momentum for optimizer ,default: 0.9
    --label_smoothing                 use label smooth in CE, default 0.1
    --save_summary_steps              logging interval,dafault:100
    --log_dir                         path to save checkpoint and log,default: ./model_1p
    --log_name                        name of log file,default: alexnet_training.log
    --save_checkpoints_steps          the interval to save checkpoint,default: 1000
    --mode                            mode to run the program (train, evaluate), default: train
    --checkpoint_dir                  path to checkpoint for evaluation,default : None
    --max_train_steps                 max number of training steps ,default : 100
    --synthetic                       whether to use synthetic data or not,default : False
    --version                         weight initialization for model,default : he_uniorm
    --do_checkpoint                   whether to save checkpoint or not, default : True
    --rank_size                       number of npus to use, default : 1
```

## 训练过程 <a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为curpath/output/ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。

