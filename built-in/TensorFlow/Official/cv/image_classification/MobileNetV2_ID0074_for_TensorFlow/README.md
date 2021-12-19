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

**修改时间（Modified） ：2020.10.14**

**大小（Size）：47M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的MobileNetV2图像分类网络训练代码**



<h2 id="概述.md">概述</h2>

MobileNetV2是一种轻量型的适用于移动端的网络，其主要是由depthwise separable，linear bottlenecks，以及inverted residuals构成。MobileNetV2作为一种轻量级backbone，被广泛应用在分类，目标检测，实例分割等计算机视觉任务中。
-   参考论文：

    [Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.](https://arxiv.org/abs/1801.04381)

-   参考实现：

    [https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
        

-   适配昇腾 AI 处理器的实现：
    
    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/MobileNetV2_ID0074_for_TensorFlow    
    
-   通过Git获取对应commit\_id的代码方法如下：
    
      
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

    

## 默认配置<a name="section91661242121611"></a>

-   数据集预处理（以ImageNet2012数据集为例，仅作为用户参考示例）：
	for training:
    -   Convert DataType and RandomResizeCrop
    -   RandomHorizontalFlip, prob=0.5
	-	Subtract with 0.5 and multiply with 2.0
	for inference:
	-	Convert dataType
	-	CenterCrop 87.5% of the original image and resize to (224,224)
	-	Subtract with 0.5 and multiply 2.0
-   训练数据集预处理（当前代码以ImageNet验证集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

-   测试数据集预处理（当前代码以ImageNet验证集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224（将图像最小边缩放到256，同时保持宽高比，然后在中心裁剪图像）
    -   根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

-   训练超参：
    -   Batch size: 256
    -   Momentum: 0.9
    -   LR scheduler: cosine annealing
    -   Learning rate\(LR\): 0.8
    -   Weight decay: 0.00004
    -   Label smoothing: 0.1
    -   Train epoch: 300
    -	Warmup_epoch: 5
    -	Optimizer: MomentumOptimizer
    -	Moving average decay=0.9999



## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，具体的参数在estimator_impl.py脚本中（脚本位于MobileNetV2_for_TensorFlow目录下）。设置precision_mode参数的脚本参考如下。


```
run_config = NPURunConfig(
             hcom_parallel=True,
             precision_mode="allow_mix_precision",
             enable_data_pre_proc=True,
             save_checkpoints_steps=self.env.calc_steps_per_epoch(),
             session_config=self.estimator_config,
             model_dir=logdir,
             iterations_per_loop=config['iterations_per_loop'],
             keep_checkpoint_max=5
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

## 数据集准备<a name="section361114841316"></a>

1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。


## 模型训练<a name="section715881518135"></a>

1.  单击“立即下载”，并选择合适的下载方式下载源码包。
2.  检查目录下是否有存在8卡IP的json配置文件“8p.json”，请用户根据实际数据配置`device_ip`参数。
    8P配置文件示例。


```
{
 "group_count": "1",
 "group_list": [
  {
   "group_name": "worker",
   "device_count": "8",
   "instance_count": "1",
   "instance_list": [
    {
     "devices":[
      {"device_id":"0","device_ip":"192.168.100.101"},
      {"device_id":"1","device_ip":"192.168.101.101"},
      {"device_id":"2","device_ip":"192.168.102.101"},
      {"device_id":"3","device_ip":"192.168.103.101"},
      {"device_id":"4","device_ip":"192.168.100.100"},
      {"device_id":"5","device_ip":"192.168.101.100"},
      {"device_id":"6","device_ip":"192.168.102.100"},
      {"device_id":"7","device_ip":"192.168.103.100"}
     ],
     "pod_name":"ascend8p",
     "server_id":"127.0.0.1"
    }
   ]
  }
 ],
 "status": "completed"
}
```

-   开始训练。
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    2. 单卡训练
       
        2.1 在脚本train_1p.sh中（脚本位于MobileNetV2_for_TensorFlow/train_1p.sh），配置`dataset_dir`训练数据集路径参数，请用户根据实际路径配置，参考示例如下。

            `--dataset_dir=/opt/npu/slimImagenet` 
    

        2.2 单卡训练训练指令（脚本位于MobileNetV2_for_TensorFlow/run_1p.sh）
        
            `bash run_1p.sh`

    1. 8卡训练
       
        2.1 在脚本train_8p.sh中（脚本位于MobileNetV2_for_TensorFlow/train_8p.sh），配置`dataset_dir`训练数据集路径参数，请用户根据实际路径配置，参考示例如下。

            `--dataset_dir=/opt/npu/slimImagenet` 
    

        2.2 单卡训练训练指令（脚本位于MobileNetV2_for_TensorFlow/run_8p.sh）
        
            `bash run_8p.sh`


-   验证。

    1. 训练完成之后，可以开始测试，修改eval_image_classifier_mobilenet.py脚本中checkpoint的文件路径以及dataset_dir路径，请用户根据实际路径配置，示例如下。

       
        ```
        --checkpoint_path=path/to/checkpoint 
        --dataset_dir=path/to/validaton
        ```


    2. 执行测试指令
    
        `python3 eval_image_classifier_mobilenet.py --checkpoint_path=path/to/checkpoint --dataset_dir=path/to/validaton`

<h2 id="迁移学习指导.md">迁移学习指导</h2>

-   数据集准备。

    数据集要求如下：

    1.1 获取数据。

    1.2 如果要使用自己的数据集，请参见“数据集准备”，需要将数据集转化为tfrecord格式。

    1.3 准确标注类别标签的数据集。

    1.4 数据集每个类别所占比例大致相同。

    1.5 数据集文件结构，请用户自行参照tfrecord脚本生成train/eval使用的TFRecord文件，包含训练集和验证集两部分，目录参考：

        ```
        |--|imagenet_tfrecord
        |   train-00000-of-01024
        |   train-00001-of-01024
        |   train-00002-of-01024
        |   ...
        |   validation-00000-of-00128
        |   validation-00000-of-00128
        |   ...
        ```


-   模型修改。

    1. 模型分类类别修改。

        1.1 使用自有数据集进行分类，如需将分类类别修改为10。修改nets/inception_resnet_v2.py将num_classes=1001设置为num_classes=10。

        ```
        def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
        dropout_keep_prob=0.8,
        reuse=None,
        scope='InceptionResnetV2',
        create_aux_logits=True,
        activation_fn=tf.nn.relu):
        ```

        1.2 修改nets/mobilenet/mobilenet_v2.py，将num_classes=1001设置为num_classes=10。

        ```
        def mobilenet(input_tensor,
        num_classes=1001,
        depth_multiplier=1.0,
        scope='MobilenetV2',
        conv_defs=None,
        finegrain_classification_mode=False,
        ```

        1.3 修改nets/inception_v2.py，将num_classes=1000设置为num_classes=10。


        ```
        def inception_v2(inputs,
        num_classes=1000,
        is_training=True,
        dropout_keep_prob=0.8,
        min_depth=16,
        depth_multiplier=1.0,
        prediction_fn=slim.softmax,
        ```



        1.4 修改estimator_impl.py，将num_classes=1001设置为num_classes=10。
    
        ```
        def model_fn(self, features, labels, mode, params):
        num_classes = 1001
        ```
    
        1.5 修改nets/post_training_quantization.py，将num_classes=1001设置为num_classes=10。
    
        ```
        flags.DEFINE_integer("num_classes", 1001,
                             "Number of output classes for the model.")
        ```
    
        1.6 修改datasets/imagenet.py，将_NUM_CLASSES=1001设置为_NUM_CLASSES=10。
    
        `_NUM_CLASSES = 1001`
    
    2. 加载预训练模型。
    
       配置文件增加参数
    
        2.1 修改文件train.py（具体配置文件名称，用户根据自己实际名称设置），增加以下参数。


​            
​            #用户根据预训练的实际ckpt进行配置
​            tf.app.flags.DEFINE_string(
​            'restore_path', '/code/ckpt/model.ckpt-187500', 'The directory where the ckpt files are stored.')
​            #不加载预训练网络中FC层权重
​            tf.app.flags.DEFINE_list(
​            'restore_exclude', ['MobilenetV2/Logits/'], 'The directory where the fc files are stored.')


        2.2 模型加载修改，修改文件estimator_impl.py，增加以下代码行。


​            
​            estimator_spec = tf.estimator.EstimatorSpec(
​                    mode=tf.estimator.ModeKeys.TRAIN, loss=total_loss, train_op=train_op)
​            #restore ckpt for finetune，
​            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=self.env.FLAGS.restore_exclude)
​            tf.train.init_from_checkpoint(self.env.FLAGS.restore_path,{v.name.split(':')[0]: v for v in variables_to_restore})   


​    
-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>


脚本和示例代码

```
├── train.py                                          //网络训练与测试代码
├── env.py                                            //超参配置
├── README.md                                         //说明文档
├── logger.py
├── eval_image_classifier_mobilenet.py                //测试脚本
├── dataloader
│    ├──data_provider.py                             //数据加载入口脚本 
├── estimator_impl.py
```


## 脚本参数<a name="section6669162441511"></a>


```
--dataset_dir              数据集路径，默认：/opt/npu/slimImagenet
--max_train_steps          最大的训练step 数， 默认：None
--iterations_per_loop      NPU运行时，device端下沉次数，默认：None
--model_name               网络模型，默认：mobilenet_v2_140
--moving_average_decay     滑动平均的衰减系数， 默认：None
--label_smoothing          label smooth 系数， 默认：0.1
--preprocessing_name       预处理方法， 默认：inception_v2
--weight_decay             正则化系数，默认：0
--batch_size               每个NPU的batch size， 默认：256
--learning_rate_decay_type 学习率衰减的策略， 默认：fixed
--learning_rate            学习率， 默认：0.1
--optimizer                优化器， 默认：sgd
--momentum                 动量， 默认：0.9 
--warmup_epochs            学习率线性warmup 的epoch数， 默认：5
--max_epoch                训练epoch次数，默认：300
```


## 训练过程<a name="section1589455252218"></a>

1. 配置8卡训练脚本（train_8p.sh）的参数，详细可参考“快速上手”。checkpoint和log文件默认保存在“result/8p/”下面，以下为log信息示例。

```
2020-06-28 15:20:50.593892: I tf_adapter/kernels/geop_npu.cc:780] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp10_0, ret_status:success ,tf session: directdbe078fad5c67345 ,graph id: 61 [0 ms]
2020-06-28 15:20:50.594560: I tf_adapter/kernels/geop_npu.cc:65] BuildOutputTensorInfo, num_outputs:1
2020-06-28 15:20:50.594604: I tf_adapter/kernels/geop_npu.cc:94] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281463098667072, output281463099285952
2020-06-28 15:20:50.594621: I tf_adapter/kernels/geop_npu.cc:758] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp10_0[ 827us]
step:73750  epoch:117.89251518342262 ips:11012.314306960132 loss:2.76953125  total_loss:3.094825029373169  lr:0.5281736254692078, train_accuracy:0.5798249840736389
I0628 15:20:50.594920 281472846090256 logger.py:42] step:73750  epoch:117.89251518342262 ips:11012.314306960132 loss:2.76953125  total_loss:3.094825029373169  lr:0.5281736254692078, train_accuracy:0.5798249840736389
INFO:tensorflow:loss = 3.094825, step = 73750
I0628 15:20:50.595537 281472846090256 basic_session_run_hooks.py:262] loss = 3.094825, step = 73750
2020-06-28 15:20:50.595977: I tf_adapter/kernels/geop_npu.cc:555] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp10_0, num_inputs:0, num_outputs:1
2020-06-28 15:20:50.596056: I tf_adapter/kernels/geop_npu.cc:423] [GEOP] tf session directdbe078fad5c67345, graph id: 61 no need to rebuild
2020-06-28 15:20:50.596076: I tf_adapter/kernels/geop_npu.cc:766] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp10_0 ,tf session: directdbe078fad5c67345 ,graph id: 61
2020-06-28 15:20:50.596142: I tf_adapter/kernels/geop_npu.cc:780] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp10_0, ret_status:success ,tf session: directdbe078fad5c67345 ,graph id: 61 [0 ms]
2020-06-28 15:20:50.597643: I tf_adapter/kernels/geop_npu.cc:65] BuildOutputTensorInfo, num_outputs:1
2020-06-28 15:20:50.597666: I tf_adapter/kernels/geop_npu.cc:94] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:281463098682112, output281463099635088
2020-06-28 15:20:50.597679: I tf_adapter/kernels/geop_npu.cc:758] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp10_0[ 1603us]
INFO:tensorflow:global_step...73750
```

## 推理/验证过程<a name="section1465595372416"></a>

1. 通过“快速上手”中的测试指令启动测试。
2. 当前只能针对该工程训练出的checkpoint进行推理测试。
3. 推理脚本参数checkpoint_path可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理，也可以是某个checkpoint的路径，默认读取“result/8p/0/results/”下面最新的文件。
4. 测试结束后会打印验证集的top1 accuracy，如下所示。


```
2020-06-26 09:15:46.236574: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xaaab02f1d790 initialized for platform Host (this does not guarantee that XLA       will be used). Devices:
2020-06-26 09:15:46.236626: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From eval_image_classifier_mobilenet.py:159: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

W0626 09:15:46.409178 281473216393232 module_wrapper.py:139] From eval_image_classifier_mobilenet.py:159: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
WARNING:tensorflow:From eval_image_classifier_mobilenet.py:160: The name tf.local_variables_initializer is deprecated. Please use tf.compat.v1.local_variables_initializer instead.
W0626 09:15:47.034382 281473216393232 module_wrapper.py:139] From eval_image_classifier_mobilenet.py:160: The name tf.local_variables_initializer is deprecated. Please use tf.compat.v1.local_variables_initializer instead.
INFO:tensorflow:Restoring parameters from /opt/npu/mobilenetv2_v1.1/result/8p/0/results/model.ckpt-125000
I0626 09:15:47.133081 281473216393232 saver.py:1284] Restoring parameters from /opt/npu/mobilenetv2_v1.1/result/8p/0/results/model.ckpt-125000
WARNING:tensorflow:From eval_image_classifier_mobilenet.py:164: The name tf.train.write_graph is deprecated. Please use tf.io.write_graph instead.
W0626 09:15:47.560457 281473216393232 module_wrapper.py:139] From eval_image_classifier_mobilenet.py:164: The name tf.train.write_graph is deprecated. Please use tf.io.write_graph instead.
0, _metric_update_op: [0.74609375]
1, _metric_update_op: [0.7480469]
2, _metric_update_op: [0.7447917]
3, _metric_update_op: [0.74121094]
4, _metric_update_op: [0.73359376]
5, _metric_update_op: [0.7317708]
6, _metric_update_op: [0.72935265]
7, _metric_update_op: [0.7260742]
8, _metric_update_op: [0.7204861]
9, _metric_update_op: [0.721875]

```

