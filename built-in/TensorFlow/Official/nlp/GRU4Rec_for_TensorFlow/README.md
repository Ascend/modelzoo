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

**修改时间（Modified） ：2021.08.04**

**大小（Size）：112K**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的GRU网络训练代码**

<h2 id="概述.md">概述</h2>

这是GRu4Rec的 TensorFlow 实现.gru4rec_BPTT 下的代码使用 BPTT 来训练 RNN。这通常比原来的 gru4rec 性能更好。
gru4rec_BP 下的代码仅使用反向传播来训练 RNN，这是我们所采用的优化方法。

-   参考论文：

    http://arxiv.org/abs/1511.06939
-   参考实现：

    https://github.com/Songweiping/GRU4Rec_TensorFlow
-   适配昇腾 AI 处理器的实现：
    
    
     https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/GRU4Rec_for_TensorFlow
        

-   通过Git获取对应commit\_id的代码方法如下：
    
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    -  layers = 1
    -  rnn_size = 100
    -  n_epochs = 3
    -  batch_size = 50
    -  dropout_p_hidden = 1
    -  learning_rate = 0.001
    -  decay = 0.96
    -  decay_steps = 1e4
    -  sigma = 0
    -  init_as_normal = False
    -  reset_after_session = True
    -  session_key = 'SessionId'
    -  item_key = 'ItemId'
    -  time_key = 'Time'
    -  grad_cap = 0
    -  test_model = 2
    -  checkpoint_dir = './checkpoint'
    -  loss = 'cross-entropy'
    -  final_act = 'softmax'
    -  hidden_act = 'tanh'
    -  n_items = (- 1)


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

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

1. 模型训练使用rsc15数据集，请用户自行获取
2. 数据集的下载及处理，请用户参考”概述--> 参考实现“ 开源代码处理
3. 数据集处理后，放入模型目录下，在训练脚本gru4rec_BP/main.py 里指定数据集路径，可正常使用


## 模型训练<a name="section715881518135"></a>

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2. 单卡训练 

        2.1 配置train_full_1p.sh脚本中`data_dir`（脚本路径test/train_full_1p.sh）,请用户根据实际路径配置，数据集参数如下所示：

            --data_dir=./data/rsc15_train_full.txt

        2.2 单p指令如下:

            bash train_full_1p.sh

   
-  验证。

    1. 修改train_full_1p.sh脚本中的train参数为train=0:
    
       
        ```
        --train=0
        ```



<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    3.  数据目录结构如下：
        
        ```
        |--data
        |  |--wmt16_de_en
        |  |  |--train.tok.clean.bpe.32000
        |  |  |--newstest2014.tok.bpe.32000
        
        ```
    
-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

脚本和示例代码

```
.

├── gru4rec_BP
│   ├──evaluation.py
│   │── main.py								 	
│   │── model.py             
├── test
│   ├── env.sh
│   ├── train_full_1p.sh
│   ├── train_performance_1p.sh
├── gru4rec_BPTT
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
└── benchmark.sh
```


## 脚本参数<a name="section6669162441511"></a>

```
  --layer: Number of GRU layers. Default is 1.  
  --size: Number of hidden units in GRU model. Default is 100.   
  --epoch: Runing epochs. Default is 3.   
  --lr : Initial learning rate. Default is 0.001.   
  --train: Specify whether training(1) or evaluating(0). Default is 1.   
  --hidden_act: Activation function used in GRU units. Default is tanh.   
  --final_act: Final activation function. Default is softmax.    
  --loss: Loss functions, cross-entropy, bpr or top1 loss. Default is cross-entropy.      
  --dropout: Dropout rate. Default is 0.5.

```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡网络训练。 
2. 将训练脚本（train_full_1p.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。 
3. 模型存储路径为“${cur_path}/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中，示例如下。 

```
Tesla V100-SXM2-32GB Epoch 0 Step 1 lr: 0.001000 loss: 3.911899 Each Step time is:0.585581
```

