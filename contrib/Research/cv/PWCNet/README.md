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
- 

## 项目路径结构

```shell
PWCNet
└─ 
  ├─README.md
  ├─output_base_v2 基于squadv2微调过的albert base模型路径
  	├─checkpoint
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  	└─...
  ├─output_large_v2 基于squadv2微调过的albert base模型路径
  	├─checkpoint
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  	└─...
  ├─albert_base_v2 albert base的预训练模型
  	├─30k-clean.model
  	├─30k-clean.vocab
  	├─albert_config.json
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta

  ├─albert_large_v2 albert large的预训练模型
  	├─30k-clean.model
  	├─30k-clean.vocab
  	├─albert_config.json
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  ├─squad_v2 存放数据目录
  	├─train-v2.0.json 数据源文件
  	├─dev-v2.0.json 数据源文件
  	├─train.tfrecord 根据train-v2.0.json生成的文件
  	├─dev.tfrecord 根据dev-v2.0.json生成的文件
  	├─pred_left_file.pkl 根据dev-v2.0.json生成的文件

  ├─squad2_base.sh albert base的启动脚本
  ├─squad2_large.sh albert large的启动脚本
```

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

## 参数解释
   详情可见[Albert](https://github.com/google-research/albert)  
 
## 训练
```
python train.py
```
## 测试 
```
python pwcnet_eval_lg-6-2-multisteps-chairsthingsmix_mpisintelfinal.py
```