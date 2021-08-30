-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.08.05**

**大小（Size）：252k**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的ctpn网络文本检测网络训练代码**

<h2 id="概述.md">概述</h2>

-   CTPN是一个实现文本检测的网络，基于TensorFlow实现，参考的原始报告是使用caffe框架实现的。

-   参考论文：

    https://arxiv.org/abs/1609.03605
    
-   参考实现：

    https://github.com/eragonruan/text-detection-ctpn
    
-   适配昇腾 AI 处理器的实现：
    
    
     https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/detection/CTPN_ID0053_for_TensorFlow
        

-   通过Git获取对应commit\_id的代码方法如下：
    
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    -   初始学习率为0.00005
    -   优化器：Adam 
    -   单卡batchsize：1 
    -   总Epoch数设置为1
    -   moving_average_decay 为0.997
    
- 训练数据集：
  - 根据您的数据集修改 utils/prepare/split_label.py 中的 DATA_FOLDER 和 OUTPUT。

  - 在根目录中运行 split_label.py

    ```shell
    python ./utils/prepare/split_label.py
    ```

  - 将在 data/dataset/ 中生成准备好的数据。

  - 调整数据大小为（H，W）=（600,900）

-   测试数据集：
    -   调整数据大小为（H，W）=（600,900）

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    if FLAGS.precision_mode == "allow_mix_precision":
         custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a>5.0.2</em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1. 根据您的数据集修改 utils/prepare/split_label.py 中的 DATA_FOLDER 和 OUTPUT。

2. 在根目录中运行 split_label.py

   ```shell
   python ./utils/prepare/split_label.py
   ```

3. 将在 data/dataset/ 中生成准备好的数据。


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

-  开始训练  
   
    1.启动训练之前，首先要配置程序运行相关环境变量。

    环境变量配置信息参见：

       [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2.单卡训练 

    ​	   2.1 配置train_full_1p.sh脚本中`data_dir`（脚本路径CTPN_ID0053_for_TensorFlow/test/train_full_1p.sh）,请用户根据实际路径配置，数据集参数如下所示：

    ```
    --data_dir=/data/dataset/resized/
    ```

    ​	   2.2 单p指令如下:

    ```
    bash train_full_1p.sh
    ```
    
- 开始验证

    验证脚本为根目录下的eval.sh，${DIR_TO_CHECKPOINTS}替换为您自己的检查点文件路径，执行：

    ```
    bash eval.sh ${DIR_TO_CHECKPOINTS}
    ```

    

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    3.  数据目录结构如下：
        
        ```
        |--data
        |  |--dataset
        |  |  |
        
        ```
    
-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

脚本和示例代码

```
├── LICENSE
├── README.md
├── README_ORI.md
├── env.sh																  // 设置环境变量脚本
├── eval.sh																	// 检测脚本
├── main
│   ├── demo.py
│   ├── train.py
│   └── train_npu.py												// 训练开始脚本
├── modelzoo_level.txt
├── nets
│   ├── model_train.py
│   └── vgg.py
├── requirements.txt												// 依赖
├── run_1p.sh
├── test
│   ├── rrc_evaluation_funcs_1_1.py
│   ├── script.py
│   └── train_full_1p.sh								     // 训练脚本
└── utils
    ├── bbox
    │   ├── bbox.c
    │   ├── bbox.pyx
    │   ├── bbox_transform.py
    │   ├── make.sh
    │   ├── nms.c
    │   ├── nms.pyx
    │   └── setup.py
    ├── dataset
    │   ├── data_provider.py
    │   ├── data_util.py
    │   └── image_resize.py
    ├── prepare
    │   ├── split_label.py
    │   └── utils.py
    ├── rpn_msr
    │   ├── __init__.py
    │   ├── anchor_target_layer.py
    │   ├── bbox.py
    │   ├── config.py
    │   ├── generate_anchors.py
    │   ├── proposal_layer.py
    │   └── tf_anchor_target_layer.py
    └── text_connector
        ├── __init__.py
        ├── detectors.py
        ├── other.py
        ├── text_connect_cfg.py
        ├── text_proposal_connector.py
        ├── text_proposal_connector_oriented.py
        └── text_proposal_graph_builder.py

```


## 脚本参数<a name="section6669162441511"></a>

```
  --learning_rate                 1e-5, default ./
  --max_steps                     50000, default data/
  --decay_steps	                  30000, default
  --pretrained_model_path         pretrain model path, default: data/vgg_16.ckpt    
```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。 
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。 
3. 模型存储路径为“${cur_path}/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。 

## 推理/验证过程<a name="section1465595372416"></a>

1. 在训练结束后,执行`bash eval.sh ${DIR_TO_CHECKPOINTS}` 进行验证。${DIR_TO_CHECKPOINTS}替换为您自己的检查点文件路径
    

```
 bash eval.sh ${DIR_TO_CHECKPOINTS}
 ...
 {"precision": xxx, "recall": xxx, "hmean": xxx, "AP": 0}
```