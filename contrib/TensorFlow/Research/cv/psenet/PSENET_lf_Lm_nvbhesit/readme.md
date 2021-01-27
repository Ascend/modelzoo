# PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network

### Introduction
This is a tensorflow re-implementation of [PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1806.02559).

Thanks for the author's ([@whai362](https://github.com/whai362)) awesome work!
____________________________________________________________________________________________________________________
_____________________________________________________________________________________________________________________
### The enviroment required:
1. Any version of tensorflow version > 1.0 should be ok.
2. python 2 or 3 will be ok.

### The AIarts enviroment is ok!:
1. Tensorflow version : 1.15
2. python version : 3.7

### Train dataset ruquired is ICDAR 2015 or ICDAR2017 MLT
On AIarts dataset used through sess.run process which path is : /data/dataset/storage/icdar/; However, the dataset is $icdar2015_train in PR.

### Train process:
>>> # Start train
>>> cd ./PSEnet/
>>> bash train_testcase.sh
>>> # Train_testcase.sh was tested already! The train log will be written to ./train.log, and the loss info will be 
>>> # printed to the screen! 
>>> # Used parameters as listed(which result in time cost was 618s):
>>> # --training_data_path=$icdar2015_train
>>>	# --checkpoint_path=$output
>>>	# --num_readers=24
>>>	# --input_size=512
>>>	# --max_steps=1000
>>>	# --learning_rate=0.0001
>>>	# --save_checkpoint_steps=100
>>>	# --save_summary_steps=10
### Results
| Database | Train total loss | Recall (%) | Steps | Time cost(s) | 
| -        | -                | -          | -     |-             |
| ICDAR 2015(val) | 1.1071    | --         | 1000  |618           |


### NO Test

### note：
1. Top and Bottom are informations of PSEnet.
2. Sses.run is successfully apply to PSEnet by Ming__blue(Gitee id). Email: summyflyer@163.com.
And there maybe some advice to tell him!
________________________________________________________________________________________________
________________________________________________________________________________________________
### About issues
If you encounter any issue check issues first, or you can open a new issue.

### Reference
1. http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
2. https://github.com/CharlesShang/FastMaskRCNN
3. https://github.com/whai362/PSENet/issues/15
4. https://github.com/argman/EAST

### Acknowledge
[@rkshuai](https://github.com/rkshuai) found a bug about concat features in model.py.

**If this repository helps you，please star it. Thanks.**
