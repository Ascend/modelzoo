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

**大小（Size）：100M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的EfficientNet-B0图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

EfficientNets是一系列图像分类网络，基于AutoML和Compound Scaling技术搜索得到。相比其他网络，EfficientNets在相似精度的条件下参数量明显更少、或者在相似参数量条件下精度明显更高。EfficientNet-B0是系列网路中最小的基础网络，其他较大尺度网络均基于它缩放生成。本文档描述的EfficientNet-B0是基于Pytorch实现的版本。

-   参考论文（需优化）：

    [Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019](https://arxiv.org/abs/1905.11946)

-   参考实现(需优化)：

    [https://github.com/tensorflow/tpu/tree/r1.15/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/r1.15/models/official/efficientnet)

-   适配昇腾 AI 处理器的实现：
    
    https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/EfficientNet_B0_ID0009_for_TensorFlow



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
    -   初始学习率为0.2，使用Exponential decay learning rate
    -   优化器：RMSProp
    -   单卡batchsize：256
    -   8卡batchsize：256*8
    -   总Epoch数设置为350
    -   Weight decay为1e-5，Momentum为0.9
    -   Label smoothing参数为0.1

-   训练数据集预处理（以ImageNet/Train为例，仅作为用户参考示例）：
    -   随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据平均值和标准偏差对输入图像进行归一化

-   测试数据集预处理（以ImageNet/Val为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224*224（对图像中央区域进行裁剪，然后将图像缩放到输入尺寸）
    -   根据平均值和标准偏差对输入图像进行归一化

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

相关代码示例如下：
    
    
    run_config = NPURunConfig(
            hcom_parallel=True,
           precision_mode='allow_mix_precision',
            enable_data_pre_proc=True,
            save_checkpoints_steps=FLAGS.num_train_images // (FLAGS.train_batch_size * int(os.getenv('RANK_SIZE'))),
            session_config=estimator_config,
            model_dir=FLAGS.model_dir,
            iterations_per_loop=FLAGS.iterations_per_loop,
            keep_checkpoint_max=5      ）


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

-   下载训练脚本。
-   解压下载训练脚本后，进入EfficientNet_B0_for_TensorFlow/efficientnet目录，检查目录下是否有存在8卡IP的json配置文件“8p.json”。
    
    8P的 json配置文件内容：
        
    ```
        {"group_count": "1","group_list": 
                            [{"group_name": "worker","device_count": "8","instance_count": "1", "instance_list": 
                            [{"devices":                    
                                           [{"device_id":"0","device_ip":"192.168.100.101"},
                                            {"device_id":"1","device_ip":"192.168.101.101"},
                                            {"device_id":"2","device_ip":"192.168.102.101"},
                                            {"device_id":"3","device_ip":"192.168.103.101"},
                                            {"device_id":"4","device_ip":"192.168.100.100"},      
                                            {"device_id":"5","device_ip":"192.168.101.100"},        
                                            {"device_id":"6","device_ip":"192.168.102.100"},     
                                            {"device_id":"7","device_ip":"192.168.103.100"}],                                          
        				"pod_name":"ascend8p",        
        	"server_id":"127.0.0.1"}]}],"status": "completed"}
    ```

-   开始训练    
    
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    2. 单卡训练

        2.1 设置单卡训练参数（脚本为train_1p.sh，脚本位于EfficientNet_B0_for_TensorFlow/train_1p.sh），示例如下。 

            请确保train_1p.sh脚本中的“--data_dir”修改为 用户生成的tfrecord的路径
            
             --data_dir=/data/slimImagenet

        2.2  单卡训练指令（脚本为run__1p.sh，脚本位于EfficientNet_B0_for_TensorFlow/run_1p.sh）
	    
            bash run_1p.sh 

    3. 8卡训练

        3.1 设置8卡训练参数（脚本为train_8p.sh，脚本位于EfficientNet_B0_for_TensorFlow/train_8p.sh），示例如下。

            请确保`train_8p.sh`脚本中的“--data_dir”修改为 用户生成的tfrecord的路径
        
            --data_dir=/data/slimImagenet
        
        3.2 8卡训练指令（脚本为run_8p.sh，脚本位于EfficientNet_B0_for_TensorFlow/run_8p.sh）
	
            bash run_8p.sh


-   验证。

    1. 测试的时候，需要修改脚本启动参数（脚本为test.sh，脚本位于EfficientNet_B0_for_TensorFlow/test.sh）

        修改test.sh脚本中"--data_dir"为用户生成的tfrecord路径，"--model_dir"为ckpt所在文件夹路径，请用户根据实际路径进行配置；

        
        ```
        --data_dir=/data/slimImagenet 
        --model_dir=result/8p/0/
        ```

    
    2. 上述文件修改完成之后，执行8卡测试指令

        `bash test.sh`



<h2 id="迁移学习指导.md">迁移学习指导</h2>


-   数据集准备。

    1. 获取数据。
       如果要使用自己的数据集，请参见“数据集准备”，需要将数据集转化为tfrecord格式。
       类别数可以通过训练参数中的num_label_classes来设置。
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

-   修改训练脚本。

    1.  模型分类类别修改。
	使用自有数据集进行分类，如需将分类类别修改为10。
	    1.1 修改utils.py文件，将num_classes=1000修改为10。
	
        ```
        def __init__(self,
        	model_name,
        	batch_size=1,
        	image_size=224,
        	num_classes=1000,
        	include_background_label=False):
        	"""Initialize internal variables."""
        ```

	    1.2 修改efficientnet_builder.py、tpu/efficientnet_tpu_builder.py文件，将num_classes=1000修改为10。
	
        
            global_params = efficientnet_model.GlobalParams(
            	batch_norm_momentum=0.99,
            	.  .  .     .  .   .
            	num_classes=1000,
            	width_coefficient=width_coefficient,
            	depth_coefficient=depth_coefficient,
        

	    1.3 修改edgetpu/efficientnet_edgetpu_builder.py文件，将num_classes=1001修改为10。
	
        
            global_params = efficientnet_model.GlobalParams(
            	batch_norm_momentum=0.99,
            	.  .  .     .  .   .
            	num_classes=1001,
            	width_coefficient=width_coefficient,
            	depth_coefficient=depth_coefficient,
        

	    1.4 修改main_npu.py文件。将num_label_classes=1000修改为10。
	
        
            flags.DEFINE_integer(
            	'num_label_classes', default=1000, help='Number of classes, at least 2')
        

	    1.5 将num_label_classes == 1001修改为10。

	    `include_background_label = (FLAGS.num_label_classes == 1001)`
	    
          1.6 修改export_model.py  "num_steps", 1000，修改为10。
	
        
            flags.DEFINE_integer(
            	"num_steps", 1000,
            	"Number of post-training quantization calibration steps to run.")
        


    2.  加载预训练模型。
    
        2.1 修改配置文件参数，修改main_npu.py文件，增加一下参数。


​            
​            RESTORE_PATH_DIR = '/code/efficientnet/ckpt111/model.ckpt-218750'
​            RESTORE_EXCLUDE_DIR = ['efficientnet-b0/model/head/dense/']
​            #用户根据预训练的实际ckpt进行配置
​            flags.DEFINE_string(
​            'restore_path',
​            default=RESTORE_PATH_DIR,
​            help=('restore path'))
​            #不加载预训练网络中FC层权重
​            flags.DEFINE_list(
​            'restore_exclude',
​            default=RESTORE_EXCLUDE_DIR,
​            help=('restore_exclude'))


        2.2 模型加载修改，增加以下代码行。


​            
​            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
​            	#restore ckpt for finetune，
​            	variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=FLAGS.restore_exclude)
​            	tf.train.init_from_checkpoint(FLAGS.restore_path,{v.name.split(':')[0]: v for v in variables_to_restore})



-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

脚本和示例代码：


        ├── efficientnet
        │  ├── 8p.json                    //rank table 配置文件
        │  ├── autoaugment.py             //自动数据增强
        │  ├── efficientnet_builder.py    //efficientnet网络构建
        │  ├── efficientnet_model.py      //efficientnet网络实现
        │  ├── imagenet_input.py          //处理imagenet数据集输入
        │  ├── logger.py                  //日志相关
        |  ├── main.py                    //原代码训练和测试主文件
        │  ├── main_npu.py                //网络训练和测试的主要实现
        │  ├── preprocessing.py           //数据预处理
        │  ├── run_1p.sh                  //单卡 运行脚本
        │  ├── run_8p.sh                  //8卡 运行脚本
        │  ├── test.sh                    //测试脚本
        │  ├── train_1p.sh                //单卡 配置脚本
        │  ├── train_8p.sh                //8卡 配置脚本
        │  ├── utils.py                   //部分调用函数封装
        │  ├── ...                        //其他文件



## 脚本参数<a name="section6669162441511"></a>


​    
​    --data_dir                        directory of dataset, default: FAKE_DATA_DIR
​    --model_dir                       directory where the model stored, default: None
​    --mode                            mode to run the code, default: train_and_eval
​    --train_batch_size                batch size for training, default: 2048
​    --train_steps                     max number of training steps, default: 218949
​    --iterations_per_loop             number of steps to run on devices each iteration, default: 1251
​    --model_name                      name of the model, default: efficientnet-b0
​    --steps_per_eval                  controls how often evaluation is performed, default: 6255
​    --eval_batch_size                 batch size for evaluating in eval mode, default: 1024
​    --base_learning_rate              base learning rate for each card, default: 0.016


## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。
2. 将训练脚本（train_1p.sh，train_8p.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为result/1p或者result/8p，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件result/8p/0/train_log.log中，示例如下。

    ```
    step:10625  epoch:16.98451489930665 ips:11605.053567595072 loss:3.900111675262451 lr:0.1615966111421585
    step:11250  epoch:17.983604011030568 ips:11613.630596574947 loss:4.157290935516357 lr:0.1615966111421585
    step:11875  epoch:18.98269312275449 ips:11579.42762098099 loss:4.254304885864258 lr:0.1615966111421585
    step:12500  epoch:19.98178223447841 ips:11596.318619725114 loss:4.1437296867370605 lr:0.1567487120628357
    step:13125  epoch:20.98087134620233 ips:11566.301582109538 loss:3.8893651962280273 lr:0.1567487120628357
    step:13750  epoch:21.97996045792625 ips:11601.901114580409 loss:4.080724239349365 lr:0.15204624831676483
    step:14375  epoch:22.979049569650172 ips:11603.619375562637 loss:3.877068519592285 lr:0.15204624831676483
    step:15000  epoch:23.97813868137409 ips:11596.124477112051 loss:3.8525354862213135 lr:0.15204624831676483
    step:15625  epoch:24.977227793098013 ips:11487.499874505798 loss:3.741398572921753 lr:0.1474848836660385
    step:16250  epoch:25.976316904821932 ips:11573.403581224962 loss:3.8359453678131104 lr:0.1474848836660385
    step:16875  epoch:26.975406016545854 ips:11491.700601198982 loss:4.053454399108887 lr:0.1430603265762329
    step:17500  epoch:27.974495128269773 ips:11518.066804875652 loss:4.134816646575928 lr:0.1430603265762329
    step:18125  epoch:28.973584239993695 ips:11554.22074394313 loss:3.676194667816162 lr:0.1387685388326645
    step:18750  epoch:29.972673351717614 ips:11615.812836818135 loss:3.803140163421631 lr:0.1387685388326645
    ```

## 推理/验证过程<a name="section1465595372416"></a>

1. 在350 epoch训练执行完成后，请参见“模型训练”中的测试流程，需要修改脚本启动参数（脚本为test.sh），需配置数据及路径和ckpt路径，然后执行脚本。

    `bash test.sh`

2. 该脚本会自动执行验证流程，验证结果若想输出至文档描述文件，则需修改启动脚本参数，否则输出至默认log 文件（./eval_efficientnet-b0.log）中。典型结果如下：

    ```
    INFO:tensorflow:Finished evaluation at 2020-09-23-11:04:37
    I0923 11:04:37.340999 281473837260816 evaluation.py:275] Finished evaluation at 2020-09-23-11:04:37
    INFO:tensorflow:Saving dict for global step 218750: global_step = 218750, loss = 2.1042383, top_1_accuracy = 0.7638822, top_5_accuracy = 0.9314303
    I0923 11:04:37.341526 281473837260816 estimator.py:2049] Saving dict for global step 218750: global_step = 218750, loss = 2.1042383, top_1_accuracy = 0.7638822, top_5_accuracy = 0.9314303
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 218750: result/8p/3/model.ckpt-218750
    I0923 11:04:45.626538 281473837260816 estimator.py:2109] Saving 'checkpoint_path' summary for global step 218750: result/8p/3/model.ckpt-218750
    INFO:tensorflow:Eval results: {'loss': 2.1042383, 'top_1_accuracy': 0.7638822, 'top_5_accuracy': 0.9314303, 'global_step': 218750}. Elapsed seconds: 76
    I0923 11:04:45.628353 281473837260816 main_npu.py:801] Eval results: {'loss': 2.1042383, 'top_1_accuracy': 0.7638822, 'top_5_accuracy': 0.9314303, 'global_step': 218750}. Elapsed seconds: 76
    ```

