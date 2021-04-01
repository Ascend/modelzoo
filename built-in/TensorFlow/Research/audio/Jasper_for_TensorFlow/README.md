-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：Speech Recognition**

**版本（Version）：1.1**

**修改时间（Modified） ：2020.12.28**

**大小（Size）：1536M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Jasper网络代码**

<h2 id="概述.md">概述</h2>

-  简述
    Jasper是一个端到端的ASR模型框架，它用卷积神经网络取代了传统得声学和发音模型。
-   参考论文：

    [Jason Li, Vitaly Lavrukhin, Boris Ginsburg, Ryan Leary, Oleksii Kuchaiev, Jonathan M. Cohen, Huyen Nguyen, Ravi Teja Gaddecc. Jasper: An End-to-End Convolutional Neural Acoustic Model. arXiv.1904.03288](https://arxiv.org/abs/1904.03288)

-   参考实现：

        
    [https://github.com/NVIDIA/OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq)
        


-   适配昇腾 AI 处理器的实现：
    
        
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/audio/Jasper_for_TensorFlow
        
        


-   通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

-  训练数据集预处理：（当前代码以LibriSpeech训练集为例，仅作为用户参考示例）
    -   该数据集是包含大约1000小时的英语语音的大型语料库。这些数据来自LibriVox项目的有声读物。

-   训练超参（单卡）：
    -   Batch size: 32
    -   grad_averaging:False
    -   优化器采用NovoGrad
    -   Learning rate\(LR\): 1e-5
    -   Weight decay: 0.001
    -   epsilon:1e-08
    -   Train epoch: 50


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

Ascend 910提供自动混合精度性能，可以根据内置优化策略，自动地将部分算子以fp16的精度模式进行计算，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

将NPURunConfig的precision_mode设为“allow_mix_precision”，即可开启混合精度模式。脚本中默认开启了混合精度。



```
sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
    # custom_op.parameter_map["mix_compile_mode"].b = True
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.optimizers.extend(["pruning",
                                               "function",
                                               "constfold",
                                               "shape",
                                               "arithmetic",
                                               "loop",
                                               "dependency",
                                               "layout",
                                               "memory",
                                               "GradFusionOptimizer"])
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

1. 请用户自行准备数据集，可选用LibriSpeech。

2. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   
## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
    
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 单卡训练
        
        2.1 设置单卡训练参数（脚本位于Jasper_for_TensorFlow / configs / speech2text //jasper5x3_LibriSpeech_nvgrad_masks_1p.py），示例如下。请确保下面例子中的`vocab_file`、`dataset_files`修改为用户准备数据集的实际路径。
                
                    base_params = {
                    .    .    .   
                            "vocab_file": "/data/librispeech/vocab/vocab.txt",
                    .    .    .
                    }
                    train_params = {
                    .    .    .
                            "dataset_files": [
                                "/data/librispeech/librivox-train-clean-100.csv",
                                "/data/librispeech/librivox-train-clean-360.csv",
                                "/data/librispeech/librivox-train-other-500.csv"
                            ],        
                    }
                    eval_params = {
                    .    .    .
                            "dataset_files": [
                                "/data/librispeech/librivox-dev-clean.csv", 
                            ],
                    }
                   infer_params = {
                    .    .    .
                            "dataset_files": [
                                   "/data/librispeech/librivox-test-clean.csv",
                            ],
       
        





        2.2 单卡训练指令（脚本位于Jasper_for_TensorFlow / scripts/run_1p.sh） 

            bash run_1p.sh 

    3. 8卡训练
        
        3.1 设置8卡训练参数（脚本位于Jasper_for_TensorFlow / configs / speech2text //jasper5x3_LibriSpeech_nvgrad_masks_8p.py），示例如下。请确保下面例子中的`vocab_file`、`dataset_files`修改为用户准备数据集的实际路径。
                
                    base_params = {
                    .    .    .   
                            "vocab_file": "/data/librispeech/vocab/vocab.txt",
                    .    .    .
                    }
                    train_params = {
                    .    .    .
                            "dataset_files": [
                                "/data/librispeech/librivox-train-clean-100.csv",
                                "/data/librispeech/librivox-train-clean-360.csv",
                                "/data/librispeech/librivox-train-other-500.csv"
                            ],        
                    }
                    eval_params = {
                    .    .    .
                            "dataset_files": [
                                "/data/librispeech/librivox-dev-clean.csv", 
                            ],
                    }
                   infer_params = {
                    .    .    .
                            "dataset_files": [
                                   "/data/librispeech/librivox-test-clean.csv",
                            ],
       
        





        2.2 单卡训练指令（脚本位于Jasper_for_TensorFlow / scripts/run_8p.sh） 

            bash run_8p.sh 
        
        说明：a. 在训练完成后，checkpoint将保存在`scripts/result/8p/${device_id}/jasper_log_folder/logs`路径下，训练日志保存在`scripts/result/8p/train_${device_id}.log`。
b. 每次重新训练需要清理日志目录或者重新指定日志目录，否则默认在原有基础上继续训练。


<h2 id="开始测试.md">开始测试</h2>

 - 参数配置
    1. 配置单卡测试参数（脚本位于Jasper_for_TensorFlow / configs / speech2text //jasper5x3_LibriSpeech_nvgrad_masks_1p.py），示例如下。请确保下面例子中的`logdir`参数修改为用户训练结果得到checkpoints的实际路径。
        
        ```
        base_params = {
        .    .    .   
                 "logdir": "jasper_log_folder",
        .    .    .
                            }
        ```

- 执行测试指令
    
    1. 上述文件修改完成之后，执行单卡测试指令（脚本位于脚本位于Jasper_for_TensorFlow / scripts/run_1p.sh）
        
        `bash run_eval.sh`


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>
    

```
│   README.md              
│   run.py                //网络训练与测试代码
│   requirements.txt      //相关依赖说明
├ ─ configs
│   └ ─ speech2text
│       └ ─    jasper5x3_LibriSpeech_nvgrad_masks_8p.py    //8卡参数脚本
│       └ ─    jasper5x3_LibriSpeech_nvgrad_masks_1p.py    //单卡参数脚本
├ ─ open_seq2seq
└ ─ scripts
    └ ─    8p.json      //rank table 配置文件
    └ ─    eval_1p.sh   //单卡测试脚本
    └ ─    run_1p.sh    //单卡训练脚本
    └ ─    run_8p.sh    //多卡训练脚本
    └ ─    run_eval.sh  //评测脚本
    └ ─    train_1p.sh  //单卡配置脚本
    └ ─    train_8p.sh  //8卡配置脚本
```


## 脚本参数<a name="section6669162441511"></a>

```
--model_path	 #original path of pb model,default value: ../model/jasper_infer_float32.pb
--data_dir       #parents dir of dev.json file，default value：../datasets
--output_dir	 #the output dir of preprocess of jasper bin files.default value:../datasets
--pre_process  	 #weather execute preprocess.option value:True/False，default value is: False
--post_process   #weather execute postprocess.option value:True/False，default value is: True
--batchSize	 #batch size of inference.default value is 1
```

## 训练过程<a name="section1589455252218"></a>

通过“快速上手”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持1，8P网络训练。训练脚本log样例如下（样例为8P网络训练的log）：

```
2020-10-20 22:59:46.245018: I tf_adapter/kernels/geop_npu.cc:104] BuildOutputTensorInfo, output index:405, total_bytes:4, shape:, tensor_ptr:281439481462080, output281439481304064
2020-10-20 22:59:46.245032: I tf_adapter/kernels/geop_npu.cc:104] BuildOutputTensorInfo, output index:406, total_bytes:4, shape:, tensor_ptr:281439482227648, output281439484073872
2020-10-20 22:59:46.245046: I tf_adapter/kernels/geop_npu.cc:104] BuildOutputTensorInfo, output index:407, total_bytes:8, shape:, tensor_ptr:281439476969152, output281439476590080
2020-10-20 22:59:46.245066: I tf_adapter/kernels/geop_npu.cc:572] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp25_233[ 512908us]
2020-10-20 23:00:11.606136: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 71
2020-10-20 23:00:11.606288: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 71
2020-10-20 23:00:11.606357: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 61
2020-10-20 23:00:11.606381: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 61
2020-10-20 23:00:11.606478: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 11
2020-10-20 23:00:11.606495: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 11
2020-10-20 23:00:11.606833: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 1
2020-10-20 23:00:11.606861: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 1
2020-10-20 23:00:11.607055: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 21
2020-10-20 23:00:11.607076: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 21
2020-10-20 23:00:11.607138: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 51
2020-10-20 23:00:11.607156: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 51
2020-10-20 23:00:11.607274: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 31
2020-10-20 23:00:11.607291: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 31
2020-10-20 23:00:11.607350: I tf_adapter/kernels/geop_npu.cc:215] [GEOP] GeOp start to finalize, tf session: directfea608cf02b1de31, graph_id_: 41
2020-10-20 23:00:11.607368: I tf_adapter/util/session_manager.cc:69] find ge session connect with tf session directfea608cf02b1de31
2020-10-20 23:00:13.947145: I tf_adapter/util/session_manager.cc:74] destroy ge session connect with tf session directfea608cf02b1de31 success.
2020-10-20 23:00:15.115646: I tf_adapter/util/ge_plugin.cc:213] [GePlugin] Close TsdClient and destroy tdt.
2020-10-20 23:00:15.127718: I tf_adapter/util/ge_plugin.cc:220] [GePlugin] Close TsdClient success.
2020-10-20 23:00:15.127764: I tf_adapter/kernels/geop_npu.cc:233] [GEOP] GePlugin Finalize success
2020-10-20 23:00:15.133104: I tf_adapter/kernels/geop_npu.cc:244] [GEOP] GeOp Finalize success, tf session: directfea608cf02b1de31, graph_id_: 41
```


## 推理/验证过程<a name="section1465595372416"></a>

通过“快速上手”中模型开始测试指令示例启动测试。

当前只能针对该工程训练出的checkpoint进行推理，且默认仅使用最新保存的ckpt进行推理。测试log样例如下：

```
2020-10-22 17:05:28.467802: I tf_adapter/util/session_manager.cc:69] find ge session connect with tf session direct13deb7e4b2445521
2020-10-22 17:05:28.883688: I tf_adapter/util/session_manager.cc:74] destroy ge session connect with tf session direct13deb7e4b2445521 success.
2020-10-22 17:05:29.193900: I tf_adapter/util/ge_plugin.cc:210] [GePlugin] Close TsdClient and destroy tdt.
2020-10-22 17:05:29.320977: I tf_adapter/util/ge_plugin.cc:217] [GePlugin] Close TsdClient success.
2020-10-22 17:05:29.321010: I tf_adapter/kernels/geop_npu.cc:232] [GEOP] GePlugin Finalize success
2020-10-22 17:05:29.321737: I tf_adapter/kernels/geop_npu.cc:243] [GEOP] GeOp Finalize success, tf session: direct13deb7e4b2445521, graph_id_: 1
*** Processed 1/85 batches
*** Processed 8/85 batches
*** Processed 16/85 batches
*** Processed 24/85 batches
*** Processed 32/85 batches
*** Processed 40/85 batches
*** Processed 48/85 batches
*** Processed 56/85 batches
*** Processed 64/85 batches
*** Processed 72/85 batches
*** Processed 80/85 batches
*** Processed 85/85 batches
*** Avg time per step: 0.447s
*** Avg objects per second: 119249.411
***     Validation WER:  0.2259
*** Finished evaluation
2020-10-22 17:05:31.795524: I tf_adapter/util/ge_plugin.cc:57] [GePlugin] destroy constructor begin
2020-10-22 17:05:31.795599: I tf_adapter/util/ge_plugin.cc:198] [GePlugin] Ge has already finalized.
2020-10-22 17:05:31.795617: I tf_adapter/util/ge_plugin.cc:59] [GePlugin] destroy constructor end
turing eval success

```
