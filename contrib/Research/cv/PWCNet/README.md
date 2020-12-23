# PWCNet
## 模型简介


## 结果
迁移[PWCNet](https://github.com/philferriere/tfoptflow) 到ascend910平台，使用的环境是[ModelArts](https://www.huaweicloud.com/product/modelarts.html)

使用[FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)数据集和[FlyThings3d](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)两个数据集进行预训练的模型，之后在Ascend 910平台上在[MPI Sintel](http://sintel.is.tue.mpg.de/downloads)数据集上训练，分别在MPI Sintel training clean set和MPI Sintel test set上测试结果如下：
<table>
    <tr>
        <td></td>
        <td >training clean set</td>
        <td >test set</td>
        <td colspan="5"> training details </td>
    <tr>
    <tr>
        <td></td>
        <td>Avg. EPE &#8595;</td>
        <td>Avg. EPE &#8595;</td>
        <td>Enviroment</td>
        <td>device </td>
        <td>batch size </td>
        <td>iterations </td>
        <td>lr schedule</td>
    <tr>
    <tr>
        <td>pretrained model</td>
        <td>2.60</td>
        <td></td>
        <td>TensorFlow, GPU</td>
        <td>2</td>
        <td>16</td>
        <td>1200000</td>
        <td>multi-steps</td>
    <tr>
    <tr>
        <td>Report in paper</td>
        <td>1.70</td>
        <td></td>
        <td>Caffe, GPU</td>
        <td>Unknown</td>
        <td>4</td>
        <td>Unknown</td>
        <td>multi-steps</td>
    <tr>
    <tr>
        <td>Reproduce on GPU</td>
        <td>1.76</td>
        <td></td>
        <td>TensorFlow, GPU</td>
        <td>1</td>
        <td>4</td>
        <td>200000</td>
        <td>multi-steps</td>
    <tr>
    <tr>
        <td>Reproduce on Ascend 910</td>
        <td>1.76</td>
        <td></td>
        <td>ModelArts, Ascend 910</td>
        <td>1</td>
        <td>4</td>
        <td>200000</td>
        <td>multi-steps</td>
    <tr>
</table>

## Requirements
- Tensorflow 1.15.0
- Ascend 910
- cv2
- numpy
- os
- shutil
- tqdm
- scikit-learn
- scipy


---

## 数据准备
预训练模型在
```
obs://pwcnet-final/pretrained/pwcnet.ckpt-595000.index
obs://pwcnet-final/pretrained/pwcnet.ckpt-595000.meta
obs://pwcnet-final/pretrained/pwcnet.ckpt-595000.data-00000-of-00001
obs://pwcnet-final/pretrained/checkpoint
```  

在MPI Sintel clean training set上训练好的模型在   
```
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000.index
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000.meta
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/pwcnet.ckpt-176000.data-00000-of-00001
obs://pwcnet-final/log/pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/checkpoint
```

在GPU上复现的模型在
```
obs://pwcnet-final/log/gpu-finetuned/pwcnet.ckpt-176000.data-00000-of-00001
obs://pwcnet-final/log/gpu-finetuned/pwcnet.ckpt-176000.meta
obs://pwcnet-final/log/gpu-finetuned/checkpoint
obs://pwcnet-final/log/gpu-finetuned/pwcnet.ckpt-176000.index
```
注意：数据集直接下载后解压，不需要对数据做预处理，
OBS已设置为公共读，不需要手动从OBS中下载数据集和checkpoint文件，代码中会从OBS中拷贝到本地。



## 训练
```
python pwcnet_finetune_lg-6-2-multisteps-mpisintelclean.py
       --iterations 200000 (the training iterations)
       --display 1000 (the interval steps to display loss)
       --save_path ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/ (the path to save checkpoint)
       --batch_size 4 (the batch size)
       --dataset /cache/ (the path of dataset, the dataset will be download to this folder from obs automatically)
```
## 测试 
```
python pwcnet_eval_lg-6-2-multisteps-chairsthingsmix_mpisintelclean.py
       --ckpt (the path of checkpoint)
       --obs False (optional, if it is True, download checkpoint from obs; if you need test the checkpoint from local dir, you should set it be False)
```
