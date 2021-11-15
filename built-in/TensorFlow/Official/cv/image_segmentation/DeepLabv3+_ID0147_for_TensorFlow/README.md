-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Semantic Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.06.18**

**大小（Size）：691M**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Deeplabv3+网络训练代码**

<h2 id="概述.md">概述</h2>
- Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasets, achieving the test set performance of 89.0% and 82.1% without any post-processing .

- 

  - 参考论文

      https://arxiv.org/pdf/1802.02611v3

  - 参考实现

      https://github.com/tensorflow/models/tree/master/research/deeplab

  - 适配昇腾 AI 处理器的实现(地址重新填写)：

      https://gitee.com/sireneden/modelzoo/tree/master/contrib/TensorFlow/Research/cv/deeplab-v3-plus/deeplabv3+_hw09124698


- 通过Git获取对应commit\_id的代码方法如下：

    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   初始学习率：0.0001
    -   优化器：momentum
    -   单卡batchsize：7
    -   总Step数：300000
    -   Weight decay：0.00004 
    -   Momentum：0.9
-   训练超参（单卡）：
    -   Batch size: 7
    -   Momentum: 0.9
    -   Learning rate\(LR\): 0.0001
    -   Weight decay: 0.00004
    -   Train steps: 300000
-   训练超参（8卡）：
    - Train steps: 32500
    - Learning rate\(LR\): 0.0008


## 支持特性 <a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```

<h2 id="训练环境准备.md">训练环境准备</h2>

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

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


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>
1. 模型训练使用pascal voc 2012 数据集，数据集请用户自行获取。
2. 模型训练使用xception_65_imagenet_coco预训练模型。
3. 获取数据集后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>
- 下载训练脚本。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练
    
        2.1 设置单卡训练参数（脚本位于./test/train_full_1p.sh），示例如下。
         ```
            #网络名称，同目录名称
            Network="Deeplabv3+_for_TensorFlow"
            #训练epoch
            train_epochs=0
            #训练batch_size
            batch_size=7
            #训练step
            train_steps=300000
            #学习率
            learning_rate=0.0001
        ```
        ```
        2.2 单卡训练指令（脚本位于./deeplabv3+/test/train_full_1p.sh） 
            
        ```
            bash train_full_1p.sh
        ```


    3. 多卡训练
    
        2.1 设置多卡训练参数（脚本位于./test/train_full_8p.sh），示例如下。
         ```
            #网络名称，同目录名称
            Network="Deeplabv3+_for_TensorFlow"
            #训练epoch
            train_epochs=0
            #训练batch_size
            batch_size=7
            #训练step
            train_steps=32500
            #学习率
            learning_rate=0.0008
        ```
        ```
        2.2 单卡训练指令（脚本位于./deeplabv3+/test/train_full_1p.sh） 
            
        ```
            bash train_full_8p.sh
        ```      

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  模型训练使用pascal voc 2012 数据集，数据集请用户自行获取。
    2.  模型训练使用xception_65_imagenet_coco预训练模型。。


- 模型训练。

    请参考“快速上手”章节（？？？）。


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    .
    ├─deeplabv3+_hw09124698										源码目录
    │  ├─best								经过训练后最好的结果
    │  ├─cp									
    │  │  └─xception_65_coco_pretrained		预训练模型
    │  ├─deeplab								主源码目录
    │  │  ├─core							
    │  │  ├─datasets						
    │  │  │  └─pascal_voc_seg				数据集存放目录
    │  │  │      └─tfrecord				已存放转换后的trainval、test数据集
    │  │  ├─deprecated
    │  │  ├─evaluation
    │  │  │  ├─g3doc
    │  │  │  │  └─img
    │  │  │  └─testdata
    │  │  │      ├─coco_gt
    │  │  │      └─coco_pred
    │  │  ├─g3doc
    │  │  │  └─img
    │  │  ├─testing
    │  │  │  └─pascal_voc_seg
    │  │  └─utils
    │  └─slim								
    │      ├─datasets
    │      ├─deployment
    │      ├─nets
    │      │  ├─mobilenet
    │      │  │  └─g3doc
    │      │  └─nasnet
    │      ├─preprocessing
    │      └─scripts


## 脚本参数<a name="section6669162441511"></a>

```
        --train_crop_size="513,513" \                                    训练输入crop尺寸
        --train_batch_size=7 \                                           训练batchsize
        --dataset="pascal_voc_seg"                                       数据集种类
        --train_logdir=log_1p                                            日志目录
        --dataset_dir=${data_path}/tfrecord                              数据集目录
        --tf_initial_checkpoint                                          预加载checkpoint
```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“test/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。