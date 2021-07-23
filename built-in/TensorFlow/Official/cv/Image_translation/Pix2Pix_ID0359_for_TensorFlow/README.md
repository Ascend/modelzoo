# Pix2Pix_ID0359_for_TensorFlow

## 目录
-   [基本信息](#基本信息)
-   [概述](#概述)
-   [训练环境准备](#训练环境准备)
-   [快速上手](#快速上手)
-   [高级参考](#高级参考)

## 基本信息

-   发布者（Publisher）：Huawei
-   应用领域（Application Domain）： CV
-   版本（Version）：1.2
-   修改时间（Modified） ：2021.7.19
-   大小（Size）：687M**
-   框架（Framework）：TensorFlow 1.15.0
-   模型格式（Model Format）：ckpt
-   精度（Precision）：Mixed
-   处理器（Processor）：昇腾910
-   应用级别（Categories）：Official
-   描述（Description）：利用pix2pix2进行图像翻译训练代码

## 概述

pix2pix是将GAN应用于有监督的图像到图像翻译的经典论文，有监督表示训练数据是成对的。图像到图像翻译（image-to-image translation）是GAN很重要的一个应用方向，什么叫图像到图像翻译呢？其实就是基于一张输入图像得到想要的输出图像的过程，可以看做是图像和图像之间的一种映射（mapping），我们常见的图像修复、超分辨率其实都是图像到图像翻译的例子。
Pix2Pix_ID0359_for_TensorFlow是一个图像处理网络，为了能更好得对图像的局部做判断，利用patchGAN的结构，也就是说把图像等分成patch，分别判断每个Patch的真假，最后再取平均。作者最后说，文章提出的这个PatchGAN可以看成所以另一种形式的纹理损失或样式损失。

- 参考论文：

    https://arxiv.org/pdf/1611.07004v1.pdf

- 参考实现：

    https://github.com/affinelayer/pix2pix-tensorflow

- 适配昇腾 AI 处理器的实现：

    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/Image_translation/Pix2Pix_ID0359_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ImageNet2012训练集为例，仅作为用户参考示例）：

  请参考“概述”->“参考实现”

- 训练超参

  - Batch size: 1
  - Learning rate（LR）：0.0002
  - Train epoch: 2

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。
  
    precision_mode="allow_mix_precision"

    parser = argparse.ArgumentParser()
    parser.add_argument('--precision_mode', dest='precision_mode', default='allow_mix_precision', help='precision mode')
    
    config = tf.ConfigProto()  # 如果没有tf.ConfigProto，需要手工添加该行
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)

    if args.precision_mode == 'allow_mix_precision':
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32,
                                                                incr_every_n_steps=1000, decr_every_n_nan_or_inf=2,
                                                                decr_ratio=0.8)
    if int(os.getenv('RANK_SIZE')) == 1:
        g_optimizer = NPULossScaleOptimizer(g_optimizer, loss_scale_manager)
    else:
        g_optimizer = NPULossScaleOptimizer(g_optimizer, loss_scale_manager, is_distributed=True)
    g_optim = npu_tf_optimizer(g_optimizer).minimize(self.g_loss, var_list=self.g_vars)

<h2 id="训练环境准备">训练环境准备</h2>

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

<h2 id="快速上手">快速上手</h2>

- 数据集准备

   请参考“概述”->“参考实现”

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本train_performance_1p.sh和train_full_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_path=/datasets/facades
     ```

  2. 启动训练。

     启动单卡性能训练 （脚本为Pix2Pix_ID0359_for_TensorFlow/test/train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh --data_path=/datasets/facades
     ```
     启动单卡精度训练 （脚本为Pix2Pix_ID0359_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh --data_path=/datasets/facades
     ```

<h2 id="高级参考">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```

├── README.md                                 //代码说明文档
├── data
│    ├──cnews_loader.py                       //数据预处理
├──helper                    
│    ├──crews_group.py                       //拆分数据集
│    ├──copy_data.sh                         //复制maxcount文件
├──images
│    ├──acc_loss.png                          //CNN准确率和误差
│    ├──acc_loss_rnn.png                     //RNN准确率和误差
│    ├──cnn_architecture.png                 //CNN结构
│    ├──rnn_architecture.png                 //RNN结构
├──model.py                                  //模型代码
├──requirements                              //网络运行所需依赖
├──run_model.py                              //8卡运行启动脚本
├──main.py                                   //网络运行代码
├──test                                 
│    ├── train_performance_1p.sh             //性能运行启动脚本
│    ├── train_full_1p.sh                    //精度运行启动脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--precision_mode         NPU运行时，默认开启混合精度
--data_path              数据集路径，默认：/datasets/facades
--autotune               autotune使能，默认关闭
--batch_size             每个NPU的batch size，默认：1
--learning_rate          初始学习率，默认：0.0002
--phase                  脚本运行传参，默认是train，可选train和test
--epoch                  默认200
--train_size             默认是100000000.0
```


## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动性能或者精度训练。性能和精度通过运行不同脚本，支持性能、精度网络训练。

2.  参考脚本的模型存储路径为test/output/*，训练脚本train_*.log中可查看性能、精度的相关运行状态。

3.  测试结束后会打印网络运行时间和精度数据，在test/output/*/train_*.log中可查看相关数据。

