-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：NLP**

**版本（Version）：1.1**

**修改时间（Modified） ：2020.10.14**

**大小（Size）：2457.6M**


**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Transformer代码**

<h2 id="概述.md">概述</h2>

Transformer是一个能够高效并行训练的序列到序列模型，该模型分为编码器（Encoder）和解码器（Decoder）两个部分，主要由多头注意力（Multi-head Attention）网络和前向（FeedForward）网络组成， 同时集成了位置编码（Position Encoding）、残差连接（Residual Connection）、层归一化（Layer Normalization）等多种技术。相比循环和卷积网络，注意力网络可以同时对任意两个输入位置的向量进行运算，提高了计算效率。 在包括翻译的多个自然语言处理任务上，该网络都取得了显著的提升。

-   参考论文：
    
    [Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.](https://arxiv.org/abs/1706.03762)

-   参考实现：


-   适配昇腾 AI 处理器的实现：    
    
    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/Transformer_for_TensorFlow

-   通过Git获取对应commit\_id的代码方法如下：
    
     
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        
    ```


## 默认配置<a name="section91661242121611"></a>

-   网络结构
    - 6层，隐层大小1024，残差和层归一化使用Pre-norm方式。

    - 使用方差缩放（Variance Scaling）的均匀分布进行参数初始化。

    - bias和层归一化的beta初始化为0，层归一化的gamma初始化为1。

-   训练数据集预处理（当前代码以wmt16 en-de训练集为例，仅作为用户参考示例）

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

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。_
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

1. 数据集准备请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括wmt16en-de。

2. 处理完成后将所需文件放在脚本文件夹的data目录下，如下:

    - data/train.tok.clean.bpe.32000.en 

    - data/train.tok.clean.bpe.32000.de 

    - data/vocab.bpe.32000

3. 当前提供的训练脚本中，是以wmt16en-de数据集为例，训练过程中进行数据预处理操作，请用户使用该脚本之前自行修改训练脚本中的数据集加载和预处理方法。

4. 将训练数据集转成tfrecord格式，命令如下：
   

    ```
    paste train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.all 
    
    python3 create_training_data_concat.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/train.l128.tfrecord --max_seq_length 128
    ```
    说明：配置`--output_file` 参数，请用户根据实际路径进行配置；
    
    5.将推理数据集转成tfrecord格式，命令如下：
    
    ```
    paste newstest2014.tok.clean.bpe.32000.en newstest2014.tok.clean.bpe.32000.de > test.all 
    
    python3 create_eval_data.py --input_file test.all --vocab_file vocab.bpe.32000 --output_file /path/newstest2014.l128.tfrecord --max_seq_length 128 --num_splits 1 --clip_to_max_len True
    ```
    
    
## 模型训练<a name="section715881518135"></a>

- 下载训练脚本。（单击“立即下载”，并选择合适的下载方式下载源码包。）

- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    2. 配置configs/transformer_big.yml中的网络参数，请用户根据自己需求进行配置；
    
    3. 配置train-ende.sh中`DATA_PATH`训练参数（脚本位于Transformer_for_TensorFlow目录下），请用户根据实际数据集路径进行配置，确保`DATA_PATH`下存在已准备的数据集，如下所示（仅供参考）：
               
        ```
                DATA_PATH="../wmt-ende"
                VOCAB_SOURCE=${DATA_PATH}/vocab.share
                VOCAB_TARGET=${DATA_PATH}/vocab.share
                TRAIN_FILES=${DATA_PATH}/concat128/train.l128.tfrecord-001-of-016
                for idx in {002..016}; do
                  TRAIN_FILES="${TRAIN_FILES},${DATA_PATH}/concat128/train.l128.tfrecord-${idx}-of-016"
                done
                DEV_SOURCES=${DATA_PATH}/dev2010.tok.zh
                DEV_TARGETS=${DATA_PATH}/dev2010.tok.en
        ```

    4.  单卡训练。
    
    单卡训练指令如下（脚本位于Transformer_for_TensorFlow/transformer_1p目录下）：
    
    ```
            bash transformer_1p/transformer_main_1p.sh
    ```


    5.  8卡训练。
        
        8卡训练指令如下（脚本位于Transformer_for_TensorFlow/transformer_8p目录下）：
        ```
            bash transformer_8p/transformer_8p.sh 
        ```


-  开始推理。

    1. 配置inference.sh中的参数，'DATA_PATH', 'TEST_SOURCES', 'MODEL_DIR' 和'output'请用户设置为自己的路径；
       
       
        ```
        DATA_PATH="../wmt-ende"
        TEST_SOURCES="${DATA_PATH}/tfrecord/newstest2014.l128.tfrecord"
        MODEL_DIR="file://PATH_TO_BE_CONFIGURED"
        .    .    .
        output: ./output-0603
        ```

    2. 推理生成翻译结果，该脚本生成的为词id的表现形式；

        bash inference.sh

    3. 生成翻译结果的文本形式。

        REF_DATA：newstest2014.tok.de

        EVAL_OUTPUT：inference.sh的生成的output文件

        VOCAB_FILE：vocab.share

        bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE

        例如：

        bash scripts/process_output.sh /data/wmt-ende/newstest2014.tok.de output-0603 /data/wmt-ende/vocab.share

    4. 测试BLEU值。

        4.1 multi-bleu.perl脚本可以通过 [perl script](https://githup.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) 下载；
    
        4.2 执行指令：

            perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
        
            例如：
        
            perl multi-bleu.perl /data/wmt-ende/newstest2014.tok.de.forbleu < output-0603.forbleu

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



## 模型固化

执行下面的命令做模型固化

```
./frozen_graph.sh

MODEL_DIR="./model_dir_base"  ckpt的路径
vocab_source: /home/wmt-ende/vocab.share   --- vocab文件的路径
vocab_target: /home/wmt-ende/vocab.share   --- vocab文件的路径
```

注意：

1、模型固化前需要参考 [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)设置环境变量；可以直接执行脚本

source  env.sh 使环境变量生效，环境变量的路径中，export install_path=/usr/local/Ascend 根据实际替换

2、由于Transformer模型较大，同时涉及动态shape，在线推理的图编译时间会较长；建议如果做在线推理的话，使用medium的模型端到端打通；

训练时修改transformer_main_1p.sh或者8p下对应的脚本命令修改最后的拉起命令为train-ende_medium.sh，样例脚本如下

```
$ cat transformer_main_1p.sh
#!/bin/bash
dir=`pwd`

# Autotune
export FLAG_AUTOTUNE="" #"RL,GA"
export TUNE_BANK_PATH=/home/HwHiAiUser/custom_tune_bank
export ASCEND_DEVICE_ID=0
mkdir -p $TUNE_BANK_PATH
chown -R HwHiAiUser:HwHiAiUser $TUNE_BANK_PATH

# DataDump
export FLAG_ENABLE_DUMP=False
export DUMP_PATH=/var/log/npu/dump
export DUMP_STEP="0|2"
export DUMP_MODE="all"
mkdir -p $DUMP_PATH
chown -R HwHiAiUser:HwHiAiUser $DUMP_PATH

export JOB_ID=10086
export DEVICE_ID=2
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=${dir}/new_rank_table_1p.json
cd ../
sh train-ende_medium.sh
```

3、medium的模型固化时间大概15min



## 在线推理

执行下面的命令做在线推理

```
cd onlineInference/infer
python3.7.5 transformer_online_inference.py

--model_path  原始pb模型的路径
--data_dir  数据集的路径，dev.json文件的父目录
--output_dir  经过预处理之后的输出文件路径
--pre_process  是否执行预处理，如果已经将json文件转换为bin文件，预处理这一步可以设置为False
--post_process 是否执行后处理，默认值为True;
--batchSize  在线推理的BatchSize
```

在线推理由于涉及动态shape的模型编译，时间会比较长；如果是medium的模型，完全执行完约1h