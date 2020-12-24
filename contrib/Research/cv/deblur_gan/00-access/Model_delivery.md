## 昇腾模型众筹DeblurGAN项目开发交付文档
#### 模型概述

DeblurGAN 是针对单张图片进行运动去模糊的生成对抗网络模型，网络将模糊图片作为输入，输出去模糊后的清晰图片。网络使用的模型是带有梯度补偿（Gradient Penalty）的Conditional Wasserstein GAN，并且增加了使用VGG19作为激活的感知损失(perceptual loss)。

GAN较为明显的优势是保留图像中的纹理细节，能够生成在人眼视觉感受上更加真实的图像。该模型将GAN用于去图像模糊问题。与传统的MSE或MAE作为优化目标相比，GAN催生了在视觉上难以与真实清晰图像区分开的解决方案，能够还原更精细的纹理细节。

DeblurGAN提出了单张图片去模糊任务中适合的网络结构和损失函数，并在PSNR等指标上取得了当时的SOTA(state-of-the art)水准，在GOPRO数据集的1111张测试图片上达到27.2的PSNR指标。

DeblurGAN的生成器结构如图，包含两个步长为2的卷积块，九个ResBlock和两个转置的卷积块。每个ResBlock包括卷积层，instance normalization和ReLU激活。
<p align="center">
  <img height="460" src="images/architecture.png">
</p>

DeblurGAN的损失函数主要包含生成对抗损失和感知损失。生成对抗损失采用WGAN-GP，感知损失使用生成图像和原始图像的VGG19第三个卷积层的激活输出的特征的L2距离。

#### 设计开发方案

- 评估分析模型训练迁移难易程度、工作量、开源代码&数据集可获得性等

模型迁移训练的难度主要在于：
1. 复现论文中提到的评估指标，PSNR指标在GOPRO测试数据集上达到27.2
2. 在适配ModelArts上的tensorflow版本和支持的算子
3. 熟悉ModelArts的训练流程和错误解决，解决模型迁移过程中的报错问题。

模型迁移的难度和工作量适中。

[论文](https://arxiv.org/pdf/1711.07064.pdf)中的模型采用[pyTorch框架实现](https://github.com/KupynOrest/DeblurGAN)，但在ModelArts中并不支持该框架，所以模型迁移参考了[开源的Tensorflow实现](https://github.com/dongheehand/DeblurGAN-tf)。开源数据集方面，[GOPRO数据集](https://seungjunnah.github.io/Datasets/gopro)已经开源,开放下载。

- 基于开源代码和数据集跑通CPU/GPU训练，并确认训练迁移精度目标

基于之前提到的开源代码和数据集，首先在GPU上跑通训练流程并复现论文中的模型精度。在训练过程中，我们发现模型的官方实现达不到论文的精度，在对应[issue](https://github.com/KupynOrest/DeblurGAN/issues/156)中也有其他人遇到相似的问题。虽然可以达到官方实现预训练模型的25.05PSNR指标，但是距离论文的27.2还有较大差距。因此我们修改了模型的损失函数，增加了直接针对生成图像和GT图像的L2 loss，在新的损失函数下训练达到了27.2的PSNR指标。

- 训练迁移设计方案（适配点）
 1. 模型迁移时，参考ModelArts官方提供的指南《ModelArts训练指导书》和《网络模型移植训练指南》，对Tensorflow中的`sess.run `API进行修改以适配NPU训练，修改`tf.ConfigProto`配置进行模型训练迁移。
 2. 在模型编译过程中，发现`tf.nn.conv2d_transpose`算子在`padding=’same’`参数下对与转置卷积输出形状的计算出现问题，在gitee上提出[issue](https://gitee.com/ascend/modelzoo/issues/I24YB6?from=project-issue)并为算子开发人员提提供对应模型的pb文件进行分析。最后相关算子问题得到解决，对应PR已经合并到主分支中。为了适配ModelArts中的版本，使用`pixelshuffle`操作替换掉`conv2d_transpose`操作，根据之前训练经验，该算子替换可以一定程度上提高网络性能。
 3. 此外对于原始开源实现中[tensorflow使用不规范的情况](https://github.com/dongheehand/DeblurGAN-tf/blob/master/ops.py#L23)进行修改，在tensorflow算子内部尽量避免使用非tf的函数，否则在session编译下沉执行环节会报错。修改对应的问题之后，ModelArts中基于NPU可以成功训练。

- 模型转换及推理设计方案

第二阶段交付

#### 交付结果
- 模型网络文件
https://jbox.sjtu.edu.cn/l/Ou6oBk
模型训练脚本
https://gitee.com/NALLEIN/modelzoo/blob/develop/contrib/Research/cv/deblur_gan/00-access/main.py
- 训练超参
训练数据集预处理：
在1000张GoPro训练数据集图像上(1280 8 720)按尺寸缩小了两倍，并随即裁剪成256x256大小。
训练超参（单卡）：
```
Batch size: 1
Momentum: 0.9
Learning rate(LR): 1e-4
Max epoch: 300
Patch size: 256
```
详见 https://gitee.com/NALLEIN/modelzoo/blob/develop/contrib/Research/cv/deblur_gan/00-access/main.py#L15-60
- 训练、测试数据集链接
[Publications Datasets CV | Seungjun Nah](https://seungjunnah.github.io/Datasets/gopro)
- 训练过程
作业名称：MA-00-access-11-26-14-32 | job084505aa

Training log(ModelArts 打屏日志) : https://jbox.sjtu.edu.cn/l/FoiVPY

Training log(tf.saver 日志) ：https://jbox.sjtu.edu.cn/l/yo7ezs

- 模型转换及推理过程

第二阶段交付

- 码云代码、模型文件链接

https://gitee.com/NALLEIN/modelzoo/tree/develop/contrib/Research/cv/deblur_gan

- 是否同意官方托管相关模型

allow-cache-pth/ckpt/pb/om ： True


