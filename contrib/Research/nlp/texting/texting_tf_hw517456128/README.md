# TextING

The code for ACL2020 paper Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks [(https://arxiv.org/abs/2004.13826)](https://arxiv.org/abs/2004.13826), implemented in Tensorflow on Ascend 910 environment.

# Usage 

Substitute mat_mul.py from https://gitee.com/ascend/modelzoo/issues/I28H6W

Start training and inference as:

```
python3.7 train.py \
	--dataset mr \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_DATA \
	--learning_rate 0.005 \
	--epochs 50 \
	--batch_size 1024 \
	--hidden 96
```

The reported result from the original paper is 79.8, and this implementation achieves around 79.4
