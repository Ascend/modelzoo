-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.09.23**

**大小（Size）：1126M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的高效CNN架构设计网络shufflenetv1**

<h2 id="概述.md">概述</h2>

​         目前，神经网络架构设计主要以计算复杂度的\emph{indirect} 度量，即FLOPs 为指导。然而，\emph{direct} 指标（例如速度）还取决于其他因素，例如内存访问成本和平台特性。因此，这项工作建议评估目标平台上的直接指标，而不仅仅是考虑 FLOP。基于一系列受控实验，这项工作推导出了几个实用的\ emph {指南}，用于有效的网络设计。因此，提出了一种新的架构，称为 \emph{ShuffleNet V2}。综合消融实验验证了我们的模型在速度和精度权衡方面是最先进的。
-   参考论文：

       [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

- 参考实现：

  https://github.com/weiSupreme/shufflenetv2-tensorflow

- 适配昇腾 AI 处理器的实现：


  https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/ShuffleNetV1-1.0x-group3_ID2129_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

-   训练数据集预处理（以ImageNet2012的Train数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   图像输入格式：TFRecord 随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据ImageNet2012数据集通用的平均值和标准偏差对输入图像进行归一化
-   测试数据集预处理（以ImageNet2012的Validation数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224*224 （将图像缩放到256 * 256，然后在中央区域裁剪图像）
    -   图像输入格式：TFRecord 根据ImageNet2012数据集通用的平均值和标准偏差对输入图像进行归一化

-   训练超参（单卡）：
    -   Batch size: 32    Weight decay: 0.0001 Label smoothing: 0.1 Train epoch: 150
    -   Momentum: 0.9
    -   LR scheduler: cosine
    -   Learning rate\(LR\): 0.01
    -   Optimizer: MomentumOptimizer
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 150


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |

## 混合精度训练<a name="section168064817164"></a>

混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

设置precision_mode参数的脚本参考如下。 

```
run_config = NPURunConfig( model_dir=flags_obj.model_dir, 
                            session_config=session_config, 
                            keep_checkpoint_max=5, 
                            save_checkpoints_steps=5000, 
                            enable_data_pre_proc=True, 
                            iterations_per_loop=iterations_per_loop, 
                            log_step_count_steps=iterations_per_loop, 
                            precision_mode='allow_mix_precision', 
                            hcom_parallel=True 
                        )
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

- 数据集准备
1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  启动训练之前，首先要配置程序运行相关环境变量。

   环境变量配置信息参见：

    [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
-  单卡训练
   
    1. 配置训练参数。 

       在脚本scripts/train_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：
       
         `−−datadir=/opt/npu/slimImagenet`

    2. 执行训练指令（脚本为scripts/run_1p.sh）。
       
        `bash run_1p.sh`

-  8卡训练
   
    1. 配置训练参数。 

       在脚本scripts/train_8p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：
       
         `−−datadir=/opt/npu/slimImagenet`

    2. 执行训练指令（脚本为scripts/run_8p.sh）。
    
-   验证。

    1. 测试的时候，需要修改脚本启动参数（脚本位于scripts/test.sh），配置mode为evaluate并在eval_dir中配置checkpoint文件所在路径，请用户根据实际路径配置，参数如下所示：
       
        
       
        ```
        −−mode=evaluate
        −−datadir=/opt/npu/slimImagenet
        ```

        

    2. 测试指令（脚本位于scripts/test.sh）.

        `bash test.sh`


<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到脚本参数data_dir对应目录下。参考代码中的数据集存放路径如下：

     - 训练集： /opt/npu/slimImagenet
     - 测试集： /opt/npu/slimImagenet

     训练数据集和测试数据集以文件名中的train和validation加以区分。

     数据集也可以放在其它目录，则修改对应的脚本入参data_dir即可。

  2. 准确标注类别标签的数据集。

  3. 数据集每个类别所占比例大致相同。

  4. 参照tfrecord脚本生成train/eval使用的TFRecord文件。

  5. 数据集文件结构，请用户自行制作TFRecord文件，包含训练集和验证集两部分，目录参考：

        ```
            |--|imagenet_tfrecord
            |  train-00000-of-01024
            |  train-00001-of-01024
            |  train-00002-of-01024
            |  ...
            |  validation-00000-of-00128
            |  validation-00000-of-00128
            |  ...
        ```

- 模型修改。

    1. 模型分类类别修改。 使用自有数据集进行分类，如需将分类类别修改为10，修改vgg16/model.py将depth=1000修改为depth=10。 
       
        `labels_one_hot = tf.one_hot(labels, depth=1000) `

    2. 修改vgg16/vgg.py，将1000设置为为10。

       
        ```
        #fc8
        x = tf.layers.dense(x, 1000, activation=None, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)) 
        ```

- 加载预训练模型。 

    1. 修改配置文件参数，修改train.py文件，增加以下参数。 
       
        ```
        parser.add_argument('--restore_path', default='/code/model.ckpt-100', help="""restore path""") #配置预训练ckpt路径 
        parser.add_argument('--restore_exclude', default=['dense_2'], help="""restore_exclude""") #不加载预训练网络中FC层权重
        ```

    
    2. 模型加载修改，修改超规格vgg16/model.py文件，增加以下代码行。 


        ```
        assert (mode == tf.estimator.ModeKeys.TRAIN)
        #restore ckpt for finetune
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=self.args.restore_exclude) 
        tf.train.init_from_checkpoint(self.args.restore_path,{v.name.split(':')[0]: v for v in variables_to_restore})
        ```

- 模型训练。

    参考“模型训练”中训练步骤。

- 模型评估。

    可以参考“模型训练”中训练步骤。

<h2 id="高级参考.md">高级参考</h2>

 脚本和示例代码


```
├── train.py                 //网络训练与测试代码 
├── README.md                //代码说明文档 
├── vgg16 
│ ├──vgg.py                  //网络构建 
│ ├──create_session.py       //sess参数配置 
│ ├──data_loader.py          //数据加载 
│ ├──layers.py               //计算accuracy 
│ ├──logger.py               //打印logging信息 
│ ├──model.py                //model estimator 
│ ├──train_helper.py         //ckpt排序 
│ ├──hyper_param.py          //配置学习率策略 
│ ├──trainer.py              //训练器配置 
│ ├──preprocessing.py        //数据预处理 
├── scripts 
│ ├──run_1p.sh               //单卡运行启动脚本 
│ ├──train_1p.sh             //单卡执行脚本 
│ ├──run_8p.sh               //8卡运行启动脚本 
│ ├──train_8p.sh             //8卡执行脚本 
│ ├──test.sh                 //推理运行脚本
│ ├──8p.json                 //多卡配置文件
```


## 脚本参数<a name="section6669162441511"></a>


```
--rank_size                                    使用NPU卡数量，默认：1 
--mode                                         运行模式，默认train_and_evaluate；可选：train，evaluate，参见本小结说明 
--max_train_steps                              训练次数，默认：100 
--iterations_per_loop                          NPU运行时，device端下沉次数，默认：10 
--max_epochs                                   训练epoch次数，推荐配合train_and_evaluate模式使用，默认：150 
--epochs_between_evals                         train_and_evaluate模式时训练和推理的间隔，默认：5 
--data_dir                                     数据集路径，默认：path/data 
--eval_dir                                     推理时checkpoint文件所在路径，默认：path/eval 
--dtype                                        网络输入数据类型，默认：tf.float32 
--use_nesterov                                 是否使用Nesterov，默认：True 
--label_smoothing                              label smooth系数，默认：0.1 
--weight_decay                                 权重衰减，默认：0.0001 
--batch_size                                   每个NPU的batch size，默认：32 
--lr                                           初始学习率，默认：0.01 
--T_max cosine_annealing                       学习率策略中的T_max值，默认：150 
--momentum                                     动量，默认：0.9 
--display_every                                打屏间隔，默认：1 
--log_name                                     log文件名，默认：vgg16.log 
--log_dir                                      ckpt文件存放路径，默认：./model_1p
```

说明：当前默认模式为train_and_evaluate，每训练epochs_between_evals个epoch测试1次，共训练max_epochs个epoch；可选模式：train，训练max_train_steps次；evaluate模式，对eval_dir目录下的ckpt进行测试。

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。 
2. 参考脚本的模型存储路径为results/1p或者results/8p，训练脚本log中包括如下信息。 

```
2020-06-20 12:08:46.650335: I tf_adapter/kernels/geop_npu.cc:714] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp41_0[ 298635878us] 
2020-06-20 12:08:46.651767: I tf_adapter/kernels/geop_npu.cc:511] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp33_0, num_inputs:0, num_outputs:1 2020-06-20 12:08:46.651882: I tf_adapter/kernels/geop_npu.cc:419] [GEOP] tf session directc244d6ef05380c63, graph id: 6 no need to rebuild 
2020-06-20 12:08:46.651903: I tf_adapter/kernels/geop_npu.cc:722] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp33_0 ,tf session: directc244d6ef05380c63 ,graph id: 6 
2020-06-20 12:08:46.652148: I tf_adapter/kernels/geop_npu.cc:735] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp33_0, ret_status:success ,tf session: directc244d6ef05380c63 ,graph id: 6 [0 ms] 
2020-06-20 12:08:46.654145: I tf_adapter/kernels/geop_npu.cc:64] BuildOutputTensorInfo, num_outputs:1 
2020-06-20 12:08:46.654179: I tf_adapter/kernels/geop_npu.cc:93] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281471054129792, output281471051824928 
2020-06-20 12:08:46.654223: I tf_adapter/kernels/geop_npu.cc:714] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp33_0[ 2321us] step: 35028 epoch: 7.0 FPS: 4289.5 loss: 3.773 total_loss: 4.477 lr:0.00996 
2020-06-20 12:08:46.655903: I tf_adapter/kernels/geop_npu.cc:511] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp33_0, num_inputs:0, num_outputs:1 2020-06-20 12:08:46.655975: I tf_adapter/kernels/geop_npu.cc:419] [GEOP] tf session directc244d6ef05380c63, graph id: 6 no need to rebuild 
2020-06-20 12:08:46.655993: I tf_adapter/kernels/geop_npu.cc:722] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp33_0 ,tf session: directc244d6ef05380c63 ,graph id: 6 
2020-06-20 12:08:46.656226: I tf_adapter/kernels/geop_npu.cc:735] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp33_0, ret_status:success ,tf session: directc244d6ef05380c63 ,graph id: 6 [0 ms]
2020-06-20 12:08:46.657765: I tf_adapter/kernels/geop_npu.cc:64] BuildOutputTensorInfo, num_outputs:1

```
## 推理/验证过程<a name="section1465595372416"></a>

1. 通过“模型训练”中的测试指令启动测试。 
2. 当前只能针对该工程训练出的checkpoint进行推理测试。 
3. 推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会进行验证。 
4. 测试结束后会打印验证集的top1 accuracy和top5 accuracy，如下所示。 

```
2020-06-20 12:31:56.960517: I tf_adapter/kernels/geop_npu.cc:354] [GEOP] GE Remove Graph success. tf session: direct3a0fa9fc2797f845 , graph id: 5 
2020-06-20 12:31:56.960537: I tf_adapter/util/session_manager.cc:50] find ge session connect with tf session direct3a0fa9fc2797f845 
2020-06-20 12:31:57.201046: I tf_adapter/util/session_manager.cc:55] destory ge session connect with tf session direct3a0fa9fc2797f845 success. 
2020-06-20 12:31:57.529579: I tf_adapter/kernels/geop_npu.cc:395] [GEOP] Close TsdClient. 
2020-06-20 12:31:57.724877: I tf_adapter/kernels/geop_npu.cc:400] [GEOP] Close TsdClient success. 
2020-06-20 12:31:57.724914: I tf_adapter/kernels/geop_npu.cc:375] [GEOP] GeOp Finalize success, tf session: direct3a0fa9fc2797f845, graph_id_: 5 step epoch top1 top5 loss checkpoint_time(UTC) 25020 1.0 36.212 63.23 3.85 
2020-06-20 11:58:14 30024 1.0 40.609 67.52 3.71 
2020-06-20 12:03:45 35028 1.0 43.494 70.31 3.57 
2020-06-20 12:08:50 40032 1.0 45.985 72.55 3.40 
2020-06-20 12:13:55 45036 2.0 48.612 75.00 3.20 
2020-06-20 12:18:59 Finished evaluation
```


