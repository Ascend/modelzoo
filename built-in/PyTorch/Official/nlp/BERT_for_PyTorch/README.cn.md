# bert模型(finetuning)使用说明

#### Requirements
请参考requirements.txt安装相关的依赖包

#### 数据集准备

1. 下载SQuADv1.1数据集：

```
cd data/squad
mkdir v1.1
cd v1.1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json --no-check-certificate
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json --no-check-certificate
wget https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ -O evaluate-v1.1.py
```

2. 确认数据集路径
    请确保数据集路径如下

```
---bert_for_pytorch
---data
   ---squad
      ---v1.1
         ---train-v1.1.json
         ---dev-v1.1.json
         ---evaluate-v1.1.py
```

​     

#### 预训练模型准备
1. 从以下链接下载预训练模型，并将预训练模型置于checkpoints目录下
https://ngc.nvidia.com/catalog/models/nvidia:bert_pyt_ckpt_large_pretraining_amp_lamb/files
2. 确认预训练模型路径
请确保如下路径：  

```
---bert_for_pytorch
   ---checkpoints
      ---bert_large_pretrained_amp.pt
```



#### 启动训练

##### 单卡
bash scripts/run_squad_npu_1p.sh 

##### 8卡
bash scripts/run_squad_npu_8p.sh




