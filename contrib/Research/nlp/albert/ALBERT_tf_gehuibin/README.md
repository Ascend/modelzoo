训练和预测脚本:

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只训练则注释掉--do_train

只预测则注释掉--do_predict

输入文件需要建立squad_v2文件夹

对于albert_base需要建立albert_base_v2, output_base_v2文件夹

对于albert_large需要建立albert_base_v2, output_base_v2文件夹

上述文件夹均可从

[百度网盘](https://pan.baidu.com/s/1F_8A398wefDj9woOJ71MwQ)提取码: 7taq 下载


# ALBERT
## 概述
迁移[Albert](https://github.com/google-research/albert) 到ascend910平台
使用的是albert_v2版本的预训练模型
|  | F1| EM |
| :-----| ----: | :----: |
| albert_base(Ascend) | 82.4| 79.4|
| albert_base(论文) | 82.1 | 79.3 |
| albert_large(Ascend) | 84.2 | 81.3 |
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
- 数据预训练模型以及微调后的模型均可以从 
[百度网盘](https://pan.baidu.com/s/1F_8A398wefDj9woOJ71MwQ) 提取码: 7taq 下载


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
### 训练

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只训练则注释掉--do_predict

### 预测和精度对比

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只预测和精度对比注释掉--do_train
