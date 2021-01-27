## 概述
BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。  
代码来自https://github.com/google-research/bert 迁移到NPU 昇腾910进行混合精度训练  
经过官方的[bert-base](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)模型预训练后可以达到EM:73.32603385833403 F1:76.53615646529383[ckpt](https://pan.baidu.com/s/1QiBVtNIbJGCHd1tqeZDqUQ) 提取码edjx  

## 代码路径解释
├─model&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//模型路径 为了方便ModelArts训练,预训练模型请放到这里  
│      bert_config.json&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//模型配置  
│      bert_model.ckpt.data-00000-of-00001&emsp;//预训练模型  
│      bert_model.ckpt.index&emsp;&emsp;&emsp;&emsp;//预训练模型  
│      bert_model.ckpt.meta&emsp;&emsp;&emsp;&emsp;&emsp;//预训练模型  
│      vocab.txt&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//词汇表  
│

└─squad2&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//数据集目录  
&emsp;&emsp;&emsp;dev-v2.0.json&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//验证集  
&emsp;&emsp;&emsp;evaluate-v2.0.py&emsp;&emsp;&emsp;&emsp;&emsp;//验证脚本  
&emsp;&emsp;&emsp; train-v2.0.json&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;//训练集  

## 启动参数
--model_dir&emsp;&emsp;&emsp;&emsp;&emsp;模型目录  
--output_dir&emsp;&emsp;&emsp;&emsp;输出目录  
--data_dir&emsp;&emsp;&emsp;&emsp;&emsp;训练集目录  
--init_checkpoint&emsp;&emsp;&emsp;预训练ckpt路径 为了方便ModelArts训练预训练模型放到model目录然后设置为ckpt文件名即可例如--init_checkpoint=bert_model.ckpt  
--max_seq_length&emsp;&emsp;&emsp;训练句子最大长度限制  
--do_train&emsp;&emsp;&emsp;&emsp;&emsp;是否训练  
--do_predict&emsp;&emsp;&emsp;&emsp;是否验证  
--train_batch_size&emsp;&emsp;&emsp;训练batch_size大小  
--predict_batch_size&emsp;&emsp;验证batch_size大小  
--learning_rate&emsp;&emsp;&emsp;&emsp;学习率  
--num_train_epochs&emsp;&emsp;&emsp;迭代数  
--warmup_proportion&emsp;&emsp;&emsp;学习率预热比例  
--save_checkpoints_steps&emsp;多少批次保存一次  
--iterations_per_loop&emsp;&emsp;多少批次显示一次  


## 训练过程
* 超参配置
    *  batch_size : 32
    *  learning_rate : 3e-5
    *  epochs : 3
    *  max_seq_length : 384
* 数据集使用SQuAD v2.0
### 训练
```
python ./run_squad.py \
--do_train=True \
--do_predict=False \
--model_dir=./model \
--data_dir=./squad2 \
--output_dir=./output_dir \
--train_batch_size=32 \
--learning_rate=3e-5 \
--num_train_epochs=3 \
--max_seq_length=384 \
--version_2_with_negative=True \
--iterations_per_loop=1000 \
--predict_batch_size=8 \
--save_checkpoints_steps=1000 \
--init_checkpoint=bert_model.ckpt
--null_score_diff_threshold   no answer对应的score与最优的非no answer对应score的差值大于该 threshold,模型才判断为no answer
```
### 预测
```
python ./run_squad.py \
--do_train=False \
--do_predict=True \
--model_dir=./model \
--data_dir=./squad2 \
--output_dir=./output_dir \
--train_batch_size=32 \
--learning_rate=3e-5 \
--num_train_epochs=3 \
--max_seq_length=384 \
--version_2_with_negative=True \
--iterations_per_loop=1000 \
--predict_batch_size=8 \
--save_checkpoints_steps=1000 \
--init_checkpoint=bert_model.ckpt
```
### 验证精度
```
python ./squad2/evaluate-v2.0.py ./squad2/dev-v2.0.json ./output_dir/predictions.json
```
