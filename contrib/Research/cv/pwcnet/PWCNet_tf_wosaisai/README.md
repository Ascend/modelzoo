# PWCNet
## 模型简介
根据“PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume”，PWCNet通过在网络设计中添加领域知识来改进FlowNet2。光流估计背后的基本思想是尽管从一帧到下一帧时像素的位置会发生变化，但像素的亮度仍是基本恒定的。我们可以在第一帧中的像素周围抓取一个小patch，并在第二帧中找到另一个小patch，该patch将最大化两个patch的某种相关性（如归一化互相关）。在整个第一帧上滑动patch并在第二帧中寻找对应最相关的patch，便得到了ost volume。这种技术相当鲁棒（具有颜色不变性），但计算成本很高。

为了减轻生成cost volume的开销，第一种优化方法是使用金字塔。使用较低分辨率的图像，可以在相对第二帧图像更小的第一帧图像上滑动一个更小的patch进行搜索，然后使用该信息指导金字塔的下一层进行更有针对性的搜索。可以在图像域或特征域中执行多尺度的运动估计（特征域上即指使用由卷积生成的按比例缩小的特征图）。实际上，PWCNet使用以较低分辨率估算的运动流的的上采样版本来扭曲第一帧，这样模型在金字塔的较高分辨率级别中只需要搜索较小的运动量（缩小搜索范围）。

![](./img/pwc-net.png)

![](./img/sample.png)
## 结果
迁移 [PWCNet](https://github.com/philferriere/tfoptflow) 到Ascend 910平台，使用的环境是 [ModelArts](https://www.huaweicloud.com/product/modelarts.html)

使用 [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) 数据集和 [FlyThings3d](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) 两个数据集进行预训练的模型，之后在ModelArts Ascend 910 TensorFlow平台上在 [MPI Sintel](http://sintel.is.tue.mpg.de/downloads) 数据集上训练，分别在MPI Sintel training clean set和MPI Sintel test set上测试结果如下：

<table>
    <tr>
        <td></td>
        <td >training clean set</td>
        <td colspan="6">training details</td>
    </tr>   
    <tr>
        <td></td>
        <td>Avg. EPE &#8595;</td>
        <td>Enviroment</td>
        <td>device </td>
        <td>batch size </td>
        <td>iterations </td>
        <td>lr schedule</td>
        <td>loss function</td>
    </tr>
    <tr>
        <td>pretrained model</td>
        <td>2.60</td>
        <td>TensorFlow, GPU</td>
        <td>2</td>
        <td>16</td>
        <td>600000</td>
        <td>multi-steps</td>
        <td>loss_multiscale(600K)</td>
    </tr>
    <tr>
        <td>Report in paper</td>
        <td>1.70</td>
        <td>Caffe, GPU</td>
        <td>Unknown</td>
        <td>4</td>
        <td>Unknown</td>
        <td>multi-steps</td>
        <td>loss_robust</td>
    </tr>
    <tr>
        <td>Reproduce on GPU</td>
        <td>1.76</td>
        <td>TensorFlow, GPU</td>
        <td>1</td>
        <td>4</td>
        <td>200000</td>
        <td>multi-steps</td>
        <td>loss_robust(200K)</td>
    </tr>
    <tr>
        <td>Reproduce on Ascend 910 V1</td>
        <td>1.76</td>
        <td>ModelArts, Ascend 910</td>
        <td>1</td>
        <td>4</td>
        <td>200000</td>
        <td>multi-steps</td>
        <td>loss_robust(200K)</td>
    </tr>
    <tr>
        <td>Reproduce on Ascend 910 V2</td>
        <td>1.25</td>
        <td>ModelArts, Ascend 910</td>
        <td>1</td>
        <td>4</td>
        <td>200000+60000</td>
        <td>multi-steps</td>
        <td>loss_robust(200K)+loss_multiscale(60K)</td>
    </tr>
</table>


## Requirements
- Tensorflow 1.15.0
- Ascend 910
- opencv-python
- numpy
- os
- shutil
- tqdm
- scikit-learn
- scipy

---
## 数据准备
### 预训练模型
预训练模型在
```
https://pan.baidu.com/s/1Qzk68OuCQM5-fWwVUfI3Cg 提取码：usks

或者OBS地址：
obs://pwcnet-final/pretrained/pwcnet.ckpt-595000.index
obs://pwcnet-final/pretrained/pwcnet.ckpt-595000.meta
obs://pwcnet-final/pretrained/pwcnet.ckpt-595000.data-00000-of-00001

下载到当前目录./pretrained
```  
### 数据集下载
http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
将其解压到当前目录下./dataset/文件夹

### Fintuned 模型
在MPI Sintel clean training set上，在ModelArts TensorFlow Ascend 910环境下训练好的 V1 模型在   
```
https://pan.baidu.com/s/1l38WF_Syr8zPI5sDtocDTw 提取码: 6vwq  

或者OBS地址：
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000.index
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000.meta
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000.data-00000-of-00001
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/checkpoint
```
V2模型在
```
https://pan.baidu.com/s/1VvU5VRncEKDh08bCbrbsaA 提取码: j5tf 
```

在GPU上复现的模型在
```
https://pan.baidu.com/s/1jkHmk-qvgsgSMI_QkynOHw 提取码: cu1n

或者OBS地址：
obs://pwcnet-final/log/gpu-finetuned/pwcnet.ckpt-176000.data-00000-of-00001
obs://pwcnet-final/log/gpu-finetuned/pwcnet.ckpt-176000.meta
obs://pwcnet-final/log/gpu-finetuned/checkpoint
obs://pwcnet-final/log/gpu-finetuned/pwcnet.ckpt-176000.index
```

## 训练
### 参数说明
```
--iterations 训练的iteration个数
--display 训练过程中打印loss的迭代间隔数
--save_path 保存checkpoint的文件夹路径
--batch_size 训练的batch_size大小
--dataset 数据集文件夹路径
--pretrained 预训练模型路径
--robust 布尔值，若为True，训练时使用loss_robust, 否则使用loss_multiscale
```

### 运行命令
####  Reproduce V1 版本
```
sh train_v1.sh
```
或者
```
python pwcnet_finetune_lg-6-2-multisteps-mpisintelclean.py
       --iterations 200000
       --display 1000 
       --save_path ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/
       --batch_size 4 
       --dataset ./dataset/
       --robust True
       --pretrained ./pretrained/pwcnet.ckpt-595000
```

####  Reproduce V2 版本
```
sh train_v2.sh
```
或者
##### Step-1 200K iterations
```
python pwcnet_finetune_lg-6-2-multisteps-mpisintelclean.py
       --iterations 200000
       --display 1000 
       --save_path ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/
       --batch_size 4 
       --dataset ./dataset/
       --robust True
       --pretrained ./pretrained/pwcnet.ckpt-595000
```
##### Step-2 60K iterations
```
python pwcnet_finetune_lg-6-2-multisteps-mpisintelclean.py
       --iterations 600000
       --display 100 
       --save_path ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned-step2/
       --batch_size 4 
       --dataset ./dataset/
       --robust False
       --pretrained ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000
```

## 测试 
### 参数说明
```
--dataset 数据集文件夹路径
--ckpt 需要测试的checkpoint的路径
```

### 运行命令
```
python pwcnet_eval_lg-6-2-multisteps-chairsthingsmix_mpisintelclean.py
       --dataset ./dataset/
       --ckpt <local checkpoint path>
```

## 离线推理
### 1、原始模型转PB模型
```
python offline_infer/convert_ckpt2pb.py \
        --ckpt ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned-step2/pwcnet-53000
```
--ckpt参数指定需要转换的模型路径，
转成的PB模型会保存在```offline_infer/pwcnet.pb```  
我们转换好的PB模型在 https://pan.baidu.com/s/1mOwNu8pTMG2yjQrjZsctrA 提取码: f514

### 2、PB模型转OM模型
使用HECS上的Mind-Studio转换PB模型到OM模型  
我们转换好的OM模型在 https://pan.baidu.com/s/1y7MdpSfXFc6O9mEZ_hKw_g 提取码: yt3w 

### 3、数据预处理
读取数据集中所有图片，对其进行预处理，保存在--output路径下gt和image两个文件夹
```
python offline_infer/dataprep.py \
       --dataset ./datasets/ \
       --output ./offline_infer/Bin/
```
或者从 OBS:// 中下载并解压到```./offline_infer/```目录下

### 4、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/ccl/msame)

### 5、推理性能精度测试
#### 推理性能测试
使用如下命令进行性能测试：
```
./msame --model ./pwcnet.om --output ./output/ --loop 100
```
测试结果如下：
```
[INFO] output data success
Inference average time: 1205.341410 ms
Inference average time without first time: 1205.274374 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```
Batchsize=1, input shape = [1, 2, 448, 1024, 3], 平均推理时间1205ms

#### 推理精度测试
使用OM模型推理结果，运行：
```
./offline_infer/msame --model ./offline_infer/pwcnet.om --input ./offline_infer/Bin/image/ --output ./offline_infer/Bin/res
```
所有的输出会保存在```./offline_infer/Bin/res/```目录下，
或者从 OBS:// 中下载并解压到该目录下

运行以下命令进行离线推理精度测试
```
python offline_infer/evaluation.py \ 
       --gt_path offline_infer/Bin/gt/ \
       --output_path pffinline_infer/Bin/res/ 
```
离线推理精度EPE=1.25，与在线模型精度一致