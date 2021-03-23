-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：_huawei_**

**应用领域（Application Domain）：_NLP_**

**版本（Version）：_1.1_**

**修改时间（Modified） ：_2020.10.14_**

_**大小（Size）**_**：**_2457.6M_


**框架（Framework）：_TensorFlow 1.15.0_**

**模型格式（Model Format）：_ckpt_**

**精度（Precision）：_Mixed_**

**处理器（Processor）：_昇腾910_**

**应用级别（Categories）：_Official_**

**描述（Description）：_基于TensorFlow框架的Transformer代码_**

<h2 id="概述.md">概述</h2>

Transformer是一个能够高效并行训练的序列到序列模型，该模型分为编码器（Encoder）和解码器（Decoder）两个部分，主要由多头注意力（Multi-head Attention）网络和前向（FeedForward）网络组成， 同时集成了位置编码（Position Encoding）、残差连接（Residual Connection）、层归一化（Layer Normalization）等多种技术。相比循环和卷积网络，注意力网络可以同时对任意两个输入位置的向量进行运算，提高了计算效率。 在包括翻译的多个自然语言处理任务上，该网络都取得了显著的提升。

   -   参考论文：

       [Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.](https://arxiv.org/abs/1706.03762)

    -   参考实现：

        ```
        无
        ```


    -   适配昇腾 AI 处理器的实现：
    
        ```
        无
        ```


    通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

-   _网络结构_
    - 6层，隐层大小1024，残差和层归一化使用Pre-norm方式。

    - 使用方差缩放（Variance Scaling）的均匀分布进行参数初始化。

    - bias和层归一化的beta初始化为0，层归一化的gamma初始化为1。

-   _训练数据集预处理（当前代码以wmt16 en-de训练集为例，仅作为用户参考示例）_

    - Transformer采用翻译文本输入，padding到最大长度128；

-   训练超参（单卡）：
    -   Learning rate(LR): 2.0 
    -   Batch size: 40
    -   Label smoothing: 0.1
    -   num_units: 1024
    -   num_layers: 6
    -   attention.num_heads: 16 


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>


```
run_config = NPURunConfig( 
        hcom_parallel = True, 
        enable_data_pre_proc = True, 
        keep_checkpoint_max=5, 
        save_checkpoints_steps=self.args.nsteps_per_epoch, 
        session_config = self.sess.estimator_config, 
        model_dir = self.args.log_dir, 
        iterations_per_loop=self.args.iterations_per_loop, 
        precision_mode='allow_mix_precision' 
     ）
```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  _硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。_
2.  _宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。_

    _当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。_

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

## 数据集准备<a name="section361114841316"></a>

1. 数据集准备请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括wmt16en-de。

2. 处理完成后将所需文件放在脚本文件夹的data目录下，如下:

    - data/train.tok.clean.bpe.32000.en 

    - data/train.tok.clean.bpe.32000.de 

    - data/vocab.bpe.32000

3. 当前提供的训练脚本中，是以wmt16en-de数据集为例，训练过程中进行数据预处理操作，请用户使用该脚本之前自行修改训练脚本中的数据集加载和预处理方法。

4. 将数据转成tfrecord格式，命令如下：
    

```
paste train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.all 

python create_training_data_concat.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/train.l128.tfrecord --max_seq_length 128
```

## 模型训练<a name="section715881518135"></a>

1.  下载训练脚本。（单击“立即下载”，并选择合适的下载方式下载源码包。）

2.  开始训练。

    启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息示例内容如下。
    当版本为Atlas Data Center Solution V100R020C00时，请使用以下环境变量：
    
```
#!/bin/bash \
    export install_path=/usr/local/Ascend \
    export PATH=${install_path}/nnae/latest/fwkacllib/ccec_compiler/bin:${PATH} \
    export ASCEND_OPP_PATH=${install_path}/nnae/latest/opp \
    export PYTHONPATH=$PYTHONPATH:${install_path}/nnae/latest/opp/op_impl/built-in/ai_core/tbe:${install_path}/tfplugin/latest/tfplugin/python/site-packages/:${install_path}/nnae/latest/fwkacllib/python/site-packages/hccl:${install_path}/nnae/latest/fwkacllib/python/site-packages/te:${install_path}/nnae/latest/fwkacllib/python/site-packages/topi
    export LD_LIBRARY_PATH=/usr/local/:/usr/local/lib/:/usr/lib/:${install_path}/nnae/latest/fwkacllib/lib64/:${install_path}/driver/lib64/common/:${install_path}/driver/lib64/driver/:${install_path}/add-ons
    
    当版本为Atlas Data Center Solution V100R020C10时，请使用以下环境变量：
    export install_path=/usr/local/Ascend/nnae/latest
    # driver包依赖
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH #仅容器训练场景配置
    export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
    #fwkacllib 包依赖
    export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
    export 
    PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
    export
    PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
    #tfplugin 包依赖
    export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH
    # opp包依赖export ASCEND_OPP_PATH=${install_path}/opp
``` 
3.  配置configs/transformer_big.yml中的网络参数，配置train-ende.sh中训练参数；
    
   
    
4.  单卡训练。

```

    cd transformer_1p

    bash transformer_main_1p.sh
```


5.  8卡训练。

    
```
    cd transformer_8p/ 

    bash transformer_8p.sh 
```


6.  开始推理。

    配置inference.sh中的参数，'DATA_PATH', 'TEST_SOURCES', 'MODEL_DIR' 和'output'设置为自己的路径；

    推理生成翻译结果，该脚本生成的为词id的表现形式；

    bash inference.sh

7.  生成翻译结果的文本形式。

    REF_DATA：newstest2014.tok.de

    EVAL_OUTPUT：inference.sh的生成的output文件

    VOCAB_FILE：vocab.share

    `bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE`

    例如：

    bash scripts/process_output.sh /data/wmt-ende/newstest2014.tok.de output-0603 /data/wmt-ende/vocab.share

8. 测试BLEU值。

    multi-bleu.perl脚本可以通过 [perl script](https://githup.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) 下载；
    
    ` perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu`

    例如：perl multi-bleu.perl /data/wmt-ende/newstest2014.tok.de.forbleu < output-0603.forbleu

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```

└─Transformer 
    ├─README.md 
    ├─configs
        ├─transformer_big.yml
        └─...
    ├─transformer_1p 
        |─new_rank_table_1p.json 
        └─transformer_main_1p.sh 
    ├─transformer_8p 
        ├─transformer_8p.sh 
        ├─transformer_p1 
            ├─new_rank_table_8p.json 
            └─transformer_main_p1.sh 
        ├─transformer_p2 
            ├─new_rank_table_8p.json 
            └─transformer_main_p2.sh 
        ├─... 
    ├─noahnmt 
        ├─bin 
            ├─train.py 
            ├─infer.py 
            └─... 
        ├─data 
            ├─text_input_pipeline.py 
            ├─input_pipeline.py 
            └─... 
        ├─attentions 
            ├─attention.py 
            ├─sum_attention.py 
            └─... 
        ├─decoders 
            ├─decoder.py 
            ├─attention_decoder.py 
            └─... 
        ├─encoders 
            ├─encoder.py 
            ├─transformer_encoder.py 
            └─... 
        ├─hooks 
            ├─metrics_hook.py 
            ├─train_hooks.py 
            └─... 
        ├─inference 
            ├─beam_search.py 
            ├─inference.py 
            └─... 
        ├─layers 
            ├─nmt_estimator.py 
            ├─rnn_cell.py 
            └─... 
        ├─metrics 
            ├─multi_bleu.py 
            ├─metric_specs.py 
            └─... 
        ├─utils 
            ├─trainer_lib.py 
            └─... 
        ├─models 
            ├─seq2seq_model.py 
            └─... 
            ├─__init__.py 
            ├─configurable.py 
            └─graph_module.py 
    ├─inference.sh 
    ├─new_rank_table_8p.json 
    ├─create_training_data_concat.py 
    └─train-ende.sh
```


## 脚本参数<a name="section6669162441511"></a>

```
#Training 
--config_paths                 Path for training config file. 
--model_params                 Model parameters for training the model, like learning_rate,dropout_rate. 
--metrics                      Metrics for model eval. 
--input_pipeline_train         Dataset input pipeline for training. 
--input_pipeline_dev           Dataset input pipeline for dev. 
--train_steps                  Training steps, default is 300000. 
--keep_checkpoint_max          Number of the checkpoint kept in the checkpoint dir. 
--batch_size                   Batch size for training, default is 40. 
--model_dir                    Model dir for saving checkpoint 
--use_fp16                     Whether to use fp16, default is True. 
```

## 训练过程<a name="section1589455252218"></a>

训练脚本会在训练过程中，每隔2500个训练步骤保存checkpoint，结果存储在model_dir目录中。

训练脚本同时会每个step打印一次当前loss值，以方便查看loss收敛情况，如下：


```
INFO:tensorflow:words/sec: 43.81k 
I0928 11:44:46.900814 281473395838992 wps_hook.py:60] words/sec: 43.81k 
INFO:tensorflow:loss = 10.8089695, step = 36 (0.219 sec) 
I0928 11:44:46.901059 281473395838992 basic_session_run_hooks.py:260] loss = 10.8089695, step = 36 (0.219 sec) 
2020-09-28 11:44:46.901399: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp9_0, num_inputs:0, num_outputs:1 2020-09-28 11:44:46.901443: I tf_adapter/kernels/geop_npu.cc:258] [GEOP] tf session directb287e87e429467f3, graph id: 31 no need to rebuild 
2020-09-28 11:44:46.901453: I tf_adapter/kernels/geop_npu.cc:602] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp9_0 ,tf session: directb287e87e429467f3 ,graph id: 31 
2020-09-28 11:44:46.901727: I tf_adapter/kernels/geop_npu.cc:615] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp9_0, ret_status:success ,tf session: directb287e87e429467f3 ,graph id: 31 [0 ms] 
2020-09-28 11:44:46.904749: I tf_adapter/kernels/geop_npu.cc:76] BuildOutputTensorInfo, num_outputs:1 
2020-09-28 11:44:46.904783: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281462830504320, output281453257618064 
2020-09-28 11:44:46.904805: I tf_adapter/kernels/geop_npu.cc:595] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp9_0[ 3352us] 
INFO:tensorflow:global_step...36 
I0928 11:44:46.904968 281473395838992 npu_hook.py:114] global_step...36 
2020-09-28 11:44:46.906018: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp21_0, num_inputs:10, num_outputs:8 2020-09-28 11:44:46.906123: I tf_adapter/kernels/geop_npu.cc:258] [GEOP] tf session directb287e87e429467f3, graph id: 71 no need to rebuild 
2020-09-28 11:44:46.906304: I tf_adapter/kernels/geop_npu.cc:602] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp21_0 ,tf session: directb287e87e429467f3 ,graph id: 71 
2020-09-28 11:44:46.906606: I tf_adapter/kernels/geop_npu.cc:615] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp21_0, ret_status:success ,tf session: directb287e87e429467f3 ,graph id: 71 [0 ms] 
2020-09-28 11:44:47.100919: I tf_adapter/kernels/geop_npu.cc:76] BuildOutputTensorInfo, num_outputs:8 
2020-09-28 11:44:47.100972: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:0, total_bytes:4, shape:, tensor_ptr:281454044286272, output281454046025984 
2020-09-28 11:44:47.100988: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:1, total_bytes:4, shape:, tensor_ptr:281454044286400, output281454043978208 
2020-09-28 11:44:47.100996: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:2, total_bytes:4, shape:, tensor_ptr:281454044286592, output281454046026240 
2020-09-28 11:44:47.101005: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:3, total_bytes:4, shape:, tensor_ptr:281454045161664, output281454045699456 
2020-09-28 11:44:47.101013: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:4, total_bytes:4, shape:, tensor_ptr:281454045161856, output281454045182336 
2020-09-28 11:44:47.101020: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:5, total_bytes:4, shape:, tensor_ptr:281454045161984, output281454045182208 
2020-09-28 11:44:47.101028: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:6, total_bytes:8, shape:, tensor_ptr:281454045266560, output281454043539264 
2020-09-28 11:44:47.101036: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:7, total_bytes:8, shape:, tensor_ptr:281454045266752, output281454043539552 
2020-09-28 11:44:47.101045: I tf_adapter/kernels/geop_npu.cc:595] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp21_0[ 194908us] 
2020-09-28 11:44:47.102274: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp9_0, num_inputs:0, num_outputs:1 2020-09-28 11:44:47.102332: I tf_adapter/kernels/geop_npu.cc:258] [GEOP] tf session directb287e87e429467f3, graph id: 31 no need to rebuild 
2020-09-28 11:44:47.102343: I tf_adapter/kernels/geop_npu.cc:602] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp9_0 ,tf session: directb287e87e429467f3 ,graph id: 31 
2020-09-28 11:44:47.102605: I tf_adapter/kernels/geop_npu.cc:615] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp9_0, ret_status:success ,tf session: directb287e87e429467f3 ,graph id: 31 [0 ms] 
2020-09-28 11:44:47.105650: I tf_adapter/kernels/geop_npu.cc:76] BuildOutputTensorInfo, num_outputs:1 
2020-09-28 11:44:47.105681: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281462830504512, output281453176492864 
2020-09-28 11:44:47.105695: I tf_adapter/kernels/geop_npu.cc:595] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp9_0[ 3351us] 
INFO:tensorflow:global_step/sec: 4.64619 
I0928 11:44:47.106539 281473395838992 basic_session_run_hooks.py:692] global_step/sec: 4.64619 
2020-09-28 11:44:47.107284: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp9_0, num_inputs:0, num_outputs:1 2020-09-28 11:44:47.107373: I tf_adapter/kernels/geop_npu.cc:258] [GEOP] tf session directb287e87e429467f3, graph id: 31 no need to rebuild 
2020-09-28 11:44:47.107391: I tf_adapter/kernels/geop_npu.cc:602] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp9_0 ,tf session: directb287e87e429467f3 ,graph id: 31 
2020-09-28 11:44:47.107602: I tf_adapter/kernels/geop_npu.cc:615] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp9_0, ret_status:success ,tf session: directb287e87e429467f3 ,graph id: 31 [0 ms] 
2020-09-28 11:44:47.110194: I tf_adapter/kernels/geop_npu.cc:76] BuildOutputTensorInfo, num_outputs:1 
2020-09-28 11:44:47.110217: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281462830977792, output281453202209616 
2020-09-28 11:44:47.110232: I tf_adapter/kernels/geop_npu.cc:595] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp9_0[ 2841us] 
2020-09-28 11:44:47.111610: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp19_0, num_inputs:0, num_outputs:1 2020-09-28 11:44:47.111668: I tf_adapter/kernels/geop_npu.cc:258] [GEOP] tf session directb287e87e429467f3, graph id: 61 no need to rebuild 
2020-09-28 11:44:47.111685: I tf_adapter/kernels/geop_npu.cc:602] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp19_0 ,tf session: directb287e87e429467f3 ,graph id: 61 
2020-09-28 11:44:47.111890: I tf_adapter/kernels/geop_npu.cc:615] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp19_0, ret_status:success ,tf session: directb287e87e429467f3 ,graph id: 61 [0 ms] 
2020-09-28 11:44:47.114397: I tf_adapter/kernels/geop_npu.cc:76] BuildOutputTensorInfo, num_outputs:1 
2020-09-28 11:44:47.114428: I tf_adapter/kernels/geop_npu.cc:103] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281463707345344, output281453333853216 
2020-09-28 11:44:47.114442: I tf_adapter/kernels/geop_npu.cc:595] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp19_0[ 2756us]
```

## 推理/验证过程<a name="section1465595372416"></a>

通过“快速上手”中的测试指令启动单卡或者多卡测试。单卡和多卡的配置与训练过程一致。

BLEU = 28.74, 59.5/34.3/22.2/15.0 (BP=1.000, ratio=1.029, hyp_len=66369, ref_len=64504)

