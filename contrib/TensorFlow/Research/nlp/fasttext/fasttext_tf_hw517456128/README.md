# Fasttext 
Bag of Tricks for Efﬁcient Text Classification
(Armand Joulin, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. 2017. In Proceedings of EACL.)

Python Tensorflow Implementation on Ascend 910 environment

Image Path : swr.cn-north-4.myhuaweicloud.com/ascend-share/3.3.0.alpha001_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_0306

# Datasets

AG_news

Download：https://pan.baidu.com/s/1L9eNGT_gZdTrqmaccpAHJQ 

pyee 

# Results

NPU  Top1 accuracy: 95.37  Top3 accuracy: 99.92
GPU  Top1 accuracy: 91.26  Top3 accuracy: 99.74

# Usage

```
python3 main.py
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--embedding_dim 10 \
	--num_epochs 5 \
	--batch_size 4096 \
	--dropout 0.5 \
	--top_k 3
```
