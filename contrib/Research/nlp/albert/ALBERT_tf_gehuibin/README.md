# ALBERT
## 概述
迁移[Albert](https://github.com/google-research/albert) 到ascend910平台,
迁移过程详见wike[ALBERT-Base模型昇腾910迁移过程精度问题代码定位
](https://gitee.com/ascend/modelzoo/wikis/ALBERT-Base%E6%A8%A1%E5%9E%8B%E6%98%87%E8%85%BE910%E8%BF%81%E7%A7%BB%E8%BF%87%E7%A8%8B%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E4%BB%A3%E7%A0%81%E5%AE%9A%E4%BD%8D?sort_id=3145861) 
得到的结果和论文的对比，
使用的是albert_v2版本的预训练模型
|  | F1| EM |
| :-----| ----: | :----: |
| albert_base(Ascend) | **82.4**| **79.4**|
| albert_base(论文) | 82.1 | 79.3 |
| albert_large(Ascend) | **85.5** | **82.5** |
| albert_large(论文) | 84.9 | 81.8 |

## Requirements
- Tensorflow 1.15.0.
- Ascend910
- sentencepiece

## 代码路径解释

```shell
albert
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
[百度网盘](https://pan.baidu.com/s/1m3HQcZlCJ3Pak7PpXiTIpg) 提取码: 6gqi
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

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只训练则注释掉--do_predict
训练base用的batch_size为32，训练large的batch_size为16,训练large的learning rate为1.5e-5
## 预测和精度对比

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只预测和精度对比注释掉--do_train

## msame推理
详见msame推理目录夹

[msame](https://gitee.com/gehuibin/modelzoo/tree/master/contrib/Research/nlp/albert/ALBERT_tf_gehuibin/msame)
