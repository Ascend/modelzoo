# PWCNet
## 结果
迁移[PWCNet](https://github.com/philferriere/tfoptflow) 到ascend910平台  

使用[FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)数据集和[FlyThings3d](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)两个数据集进行[预训练的模型](http://bit.ly/tfoptflow)，之后在Ascend 910平台上在MPI Sintel数据集上训练，分别在MPI Sintel training set和MPI Sintel test set上测试结果如下：
|  | AEPE  | other metrics |
| :-----| ----: | :----: |
| MPI training set (in paper) |   |   |
| MPI training set (on RTX 2080Ti GPU) | | |
| MPI training set (on Ascend 910) | | |
| MPI test set (in paper) | | |
| MPI test set (on RTX 2080Ti GPU) | | |
| MPI test set (on Ascend 910) | | |

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
- ···

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

## 准备数据和模型
数据预训练模型以及微调后的模型均可以从 
[百度网盘](https://blank) 提取码: blank  
其中squad_v2目录夹为数据  
albert_base/large_v2为预训练模型  
output_base/large_v2为微调后的模型  



## 参数解释
   详情可见[Albert](https://github.com/google-research/albert)  
  --output_dir 输出的模型路径  
  --input_dir 训练集目录  
  --model_dir 预训练模型目录  
  --do_lower_case 小写  
  --max_seq_length 最大句子长度  
  --doc_stride 跨步长度  
  --max_query_length 最大问题长度  
  --do_train 训练  
  --do_predict 预测  
  --train_batch_size 训练batch size  
  --predict_batch_size 预测batch size   
  --learning_rate 学习率  
  --num_train_epochs 迭代数  
  --warmup_proportion 训练预热比例  
  --save_checkpoints_steps 多少批次保存  
  --n_best_size 预测的个数  
  --max_answer_length 最大答案长度  
## 训练
```
python train.py
```
## 测试 
```
python pwcnet_eval_lg-6-2-multisteps-chairsthingsmix_mpisintelfinal.py
```