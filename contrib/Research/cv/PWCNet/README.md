# PWCNet
## 结果
迁移[PWCNet](https://github.com/philferriere/tfoptflow) 到ascend910平台  

使用[FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)数据集和[FlyThings3d](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)两个数据集进行预训练的模型，之后在Ascend 910平台上在[MPI Sintel](http://sintel.is.tue.mpg.de/downloads)数据集上训练，分别在MPI Sintel training clean set和MPI Sintel test set上测试结果如下：
<table>
    <tr>
        <td></td>
        <td >training clean set</td>
        <td >test set</td>
    <tr>
    <tr>
        <td></td>
        <td>Avg. EPE&#8595;</td>
        <td>Avg. EPE&#8595;</td>
    <tr>
    <tr>
        <td>pretrained model</td>
        <td>2.60</td>
        <td></td>
    <tr>
    <tr>
        <td>in paper</td>
        <td>1.70</td>
        <td></td>
    <tr>
    <tr>
        <td>on TitanXP GPU</td>
        <td></td>
        <td></td>
    <tr>
    <tr>
        <td>on Ascend 910</td>
        <td>1.76</td>
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
数据预训练模型在
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
 
# ```
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
```