-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2020.12.22**

**大小（Size）：624M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的InceptionV4网络图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

InceptionV4是2016年提出的Inception系列网络的第四个版本，随着ResNet网络的出现以及在主流数据集上的良好表现，谷歌希望将残差结构引入到Inception网络中得到更好的表现，同时注意到InceptionV3的部分结构有不必要的复杂性，于是尝试在不引入残差结构的情况下改进原来的的Inception结构，最终达到和与ResNet结合方式相同的精度。
-   参考论文：

    [Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi, ​​Inception-v4, Inception-ResNet and the Impact of Residual Connection on Learning.2016](https://arxiv.org/abs/1602.07261)
-   参考实现：

    [https://github.com/tensorflow/models/tree/master/research/slim](https://github.com/tensorflow/models/tree/master/research/slim)
-   适配昇腾 AI 处理器的实现：
    
        
     https://github.com/Ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/InceptionV4_for_TensorFlow
        

-   通过Git获取对应commit\_id的代码方法如下：
    
        
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    -   初始学习率为0.045，使用Cosine learning rate 
    -   优化器：RMSProp 
    -   单卡batchsize：128 
    -   8卡batchsize：64*8 
    -   总Epoch数设置为100 
    -   Weight decay为0.00001，Momentu为0.9，decay为0.9，epsilon为1.0 
    -   Label smoothing参数为0.1

-   训练数据集预处理（以ImageNet/Train为例，仅作为用户参考示例）：
    -   图像的输入尺寸为299*299
    -   随机中心裁剪图像
    -   对比度，饱和度，色彩变换
    -   均值为0，归一化为[-1，1] 

-   测试数据集预处理（以ImageNet/Val为例，仅作为用户参考示例）：
    -   图像的输入尺寸为299*299 
    -   均值为0，归一化为[-1，1] 

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

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

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  开始训练    
    
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://github.com/Ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2. 单卡训练 

        2.1 配置run_1p.sh脚本中`data_dir`（脚本路径InceptionV4_for_TensorFlow/script/run_1p.sh）,请用户根据实际路径配置，数据集参数如下所示：

            --data_dir=/opt/npu/imagenet_data

        2.2 单p指令如下:

            bash run_1p.sh

    3. 8卡训练  
    
        3.1 配置run_8p.sh脚本中`data_dir`（脚本路径InceptionV4_for_TensorFlow/script/run_8p.sh）,请用户根据实际路径配置，数据集参数如下所示：
            
            --data_dir=/opt/npu/imagenet_data

        3.2 8p指令如下: 
            
            bash run_8p.sh

-  验证。

    1. 将运行模式mode修改为evaluate，并配置训练ckpt路径,请用户根据实际路径进行修改,如下所示：
    
        
        ```
        --mode=evaluate  
        --data_dir=/opt/npu/imagenet_data
        ```



<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备，需要将数据集转化为tfrecord格式。类别数可以通过训练参数中的num_classes来设置。
    2.  数据集每个类别所占比例大致相同。
    3.  数据目录结构如下：
        
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

- 修改训练脚本。

    1. 模型分类类别修改。 

        1.1 使用自有数据集进行分类，如需将分类类别修改为10，修改inception/inception_v3.py文件，将num_classes=1001设置为num_classes=10。 
                        
            def inception_v3(inputs, num_classes=1001, 
                                            is_training=True, 
                                            dropout_keep_prob=0.8, 
                                            reuse=None, 
                                            scope='InceptionV4', 
                                            create_aux_logits=True): 

        1.2 修改文件inception/model.py，将depth=1000修改为depth=10，将num_classes=1000修改为num_classes=10。 
            
            
            labels_one_hot = tf.one_hot(labels, depth=1000) 
            ... ... 
            if is_training: 
                with slim.arg_scope(inception_v3.inception_v3_arg_scope(weight_decay=self.args.weight_decay)): 
                top_layer, end_points = inception_v3.inception_v3(inputs=features, num_classes=1000, dropout_keep_prob=0.8, is_training = True) 
            else: 
                with slim.arg_scope(inception_v3.inception_v3_arg_scope()): 
                top_layer, end_points = inception_v3.inception_v3(inputs=features, num_classes=1000, dropout_keep_prob=1.0, is_training = False)
            
     2. 加载预训练模型。

        2.1 修改配置文件参数，修改文件train.py，增加以下参数。 
            
            parser.add_argument('--restore_path', default='ckpt/model.ckpt-250200', help="""restore path""") #配置预训练ckpt路径 
        
        2.2 模型加载修改，修改文件inception/model.py，增加以下代码行。 
            
            
            
            assert (mode == tf.estimator.ModeKeys.TRAIN)
            #restore ckpt for finetune
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['InceptionV4/Logits/Logits','InceptionV4/AuxLogits/Aux_logits'])                             
            tf.train.init_from_checkpoint(self.args.restore_path,{v.name.split(':')[0]: v for v in variables_to_restore})
            

-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

脚本和示例代码

```
├── train.py                     //网络训练与测试代码 
├── README.md                    //说明文档 
├── inception 
│ ├──inception_v3.py             //网络构建 
│ ├──create_session.py           //session的参数 
│ ├──data_loader.py              //数据加载 
│ ├──layers.py                   //计算accuracy，loss 
│ ├──logger.py                   //logging message 
│ ├──model.py                    //tf.estimator的封装 
│ ├──train_helper.py             //helper_class 
│ ├──hyper_param.py              //hyper paramaters设置 
│ ├──trainer.py                  //estimator训练和验证 
│ ├──inception_preprocessing.py  //数据预处理
├── scripts 
│ ├──run_1p.sh                   //单卡运行脚本 
│ ├──train_1p.sh                 //单卡配置脚本 
│ ├──run_8p.sh                   //8卡运行脚本 
│ ├──train_8p.sh                 //8卡配置脚本| 
│ ├──8p.json                     //rank table配置文件 
│ ├──test.sh                     //数据测试
```


## 脚本参数<a name="section6669162441511"></a>

```
--data_dir             train data dir, default : path/to/data 
--batch_size           mini-batch size ,default: 256 
--lr initial           learning rate,default: 0.01 
--max_epochs           max epoch num to train the model:default: 100 
--weight_decay         weight decay factor for regularization loss ,default: 4e-4 
--momentum             momentum for optimizer ,default: 0.9 
--label_smoothing      use label smooth in CE, default 0.1 
--log_dir              path to save checkpoint and log,default: ./model 
--log_name             name of log file,default: googlebet.log 
--mode                 mode to run the program (train, train_and_evaluate,evaluate), default: train 
--eval_dir             path to checkpoint for evaluation,default : None 
--max_train_steps      max number of training steps ,default : 100 
--iteration_per_loop   the number of steps in devices for each iteration, default: 10 
--use_nesterov         whether to use Nesterov in optimizer. default: True 
--rank_size            number of npus to use, default : 1 
--T_max                max epochs for cos_annealing learning rate 
--epochs_between_evals set the model=train_and_evaluate, evaluation will be performed after sveral epochs, default:5

```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。 
2. 将训练脚本（train_1p.sh,train_8p.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。 
3. 模型存储路径为“results/1p”或者“results/8p”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件results/1p/0/model/inception_v3.log中，示例如下。 

```
step: 12100 epoch: 1.2 FPS: 469.5 loss: 4.676 total_loss: 5.051 lr:0.04499 
step: 12200 epoch: 1.2 FPS: 469.6 loss: 4.922 total_loss: 5.297 lr:0.04499 
step: 12300 epoch: 1.2 FPS: 469.6 loss: 4.953 total_loss: 5.328 lr:0.04499 
step: 12400 epoch: 1.2 FPS: 469.7 loss: 4.758 total_loss: 5.133 lr:0.04499 
step: 12500 epoch: 1.2 FPS: 469.6 loss: 4.957 total_loss: 5.332 lr:0.04499 
step: 12600 epoch: 1.3 FPS: 469.5 loss: 4.594 total_loss: 4.969 lr:0.04499 
step: 12700 epoch: 1.3 FPS: 469.6 loss: 4.707 total_loss: 5.082 lr:0.04499 
step: 12800 epoch: 1.3 FPS: 469.6 loss: 4.574 total_loss: 4.950 lr:0.04499 
step: 12900 epoch: 1.3 FPS: 469.5 loss: 4.809 total_loss: 5.184 lr:0.04499 
step: 13000 epoch: 1.3 FPS: 469.7 loss: 4.664 total_loss: 5.040 lr:0.04499 
step: 13100 epoch: 1.3 FPS: 469.6 loss: 4.555 total_loss: 4.930 lr:0.04499 
step: 13200 epoch: 1.3 FPS: 469.6 loss: 4.703 total_loss: 5.079 lr:0.04499 
step: 13300 epoch: 1.3 FPS: 469.6 loss: 4.543 total_loss: 4.919 lr:0.04499 
step: 13400 epoch: 1.3 FPS: 469.7 loss: 4.738 total_loss: 5.114 lr:0.04499 
step: 13500 epoch: 1.3 FPS: 469.6 loss: 4.707 total_loss: 5.083 lr:0.04499 
step: 13600 epoch: 1.4 FPS: 469.6 loss: 4.793 total_loss: 5.169 lr:0.04499 
step: 13700 epoch: 1.4 FPS: 469.6 loss: 4.520 total_loss: 4.895 lr:0.04499 
step: 13800 epoch: 1.4 FPS: 469.6 loss: 4.672 total_loss: 5.048 lr:0.04499 
step: 13900 epoch: 1.4 FPS: 469.6 loss: 4.562 total_loss: 4.939 lr:0.04499 
step: 14000 epoch: 1.4 FPS: 469.6 loss: 4.742 total_loss: 5.118 lr:0.04499 
step: 14100 epoch: 1.4 FPS: 469.5 loss: 4.555 total_loss: 4.931 lr:0.04499

```

## 推理/验证过程<a name="section1465595372416"></a>

1. 在100 epoch训练执行完成后： 
    方法一：参照“模型训练”中的测试流程，需要修改脚本启动参数（脚本位于scripts/train_1p.sh）将mode设置为evaluate，增加eval_dir的路径，然后执行脚本。 
    
    方法二：在训练过程中将mode修改为train_and_evaluate，设置epochs_between_evals=5，会训练5个epoch推理一次精度。 
    
    `bash run_1p.sh `
    
    方法一：该脚本会自动执行验证流程，验证结果会输出到 /results/*p/0/model/eval.log文件中。 
    
    方法二：会把训练和推理结果写到同一个日志文件中/results/*p/model/inception_v3.log。 

```
step: 24000 epoch: 9.6 FPS: 3109.4 loss: 2.855 total_loss: 3.670 lr:0.04411 
step: 24100 epoch: 9.6 FPS: 3108.1 loss: 2.643 total_loss: 3.455 lr:0.04411 
step: 24200 epoch: 9.7 FPS: 3108.8 loss: 3.014 total_loss: 3.825 lr:0.04411 
step: 24300 epoch: 9.7 FPS: 3108.8 loss: 3.041 total_loss: 3.851 lr:0.04411 
step: 24400 epoch: 9.8 FPS: 3107.3 loss: 3.248 total_loss: 4.056 lr:0.04411 
step: 24500 epoch: 9.8 FPS: 3108.8 loss: 3.363 total_loss: 4.170 lr:0.04411 
step: 24600 epoch: 9.8 FPS: 3108.8 loss: 3.416 total_loss: 4.221 lr:0.04411 
step: 24700 epoch: 9.9 FPS: 3109.2 loss: 2.854 total_loss: 3.658 lr:0.04411 
step: 24800 epoch: 9.9 FPS: 3109.2 loss: 3.143 total_loss: 3.945 lr:0.04411 
step: 24900 epoch: 10.0 FPS: 3108.9 loss: 3.014 total_loss: 3.815 lr:0.04411 
step: 25000 epoch: 10.0 FPS: 3109.2 loss: 3.055 total_loss: 3.855 lr:0.04411 
Starting to evaluate Validation dataset size: 49921 step epoch top1 top5 loss checkpoint_time(UTC) 25020 10.0 48.210 73.99 3.12 2020-09-02 00:40:10 Starting a training cycle Step Epoch Speed Loss FinLoss LR 
step: 25100 epoch: 10.0 FPS: 251.3 loss: 2.830 total_loss: 3.628 lr:0.04390 
step: 25200 epoch: 10.1 FPS: 3103.9 loss: 3.164 total_loss: 3.961 lr:0.04390 
step: 25300 epoch: 10.1 FPS: 3104.1 loss: 3.303 total_loss: 4.098 lr:0.04390 
step: 25400 epoch: 10.2 FPS: 3103.8 loss: 3.197 total_loss: 3.991 lr:0.04390
step: 25500 epoch: 10.2 FPS: 3104.5 loss: 2.998 total_loss: 3.791 lr:0.04390

```
