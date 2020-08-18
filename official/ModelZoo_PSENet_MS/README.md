# PSENet Example
## Description
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.
## Requirements
+ install Mindspore
+ install [pyblind11](https://github.com/pybind/pybind11)
+ install [Opencv3.4](https://docs.opencv.org/3.4.9/d7/d9f/tutorial_linux_install.html)
+ Download the dataset [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization)
+ We use ICDAR2015 as training dataset in this example by default,and you can also use your own datasets
## Example structure
.  
└── PSENet  
&emsp;├── README.md  
&emsp;├── scripts  
&emsp;&emsp;├── run_distribute_train.sh  
&emsp;&emsp;└── eval_ic15.sh  
&emsp;├── src  
&emsp;&emsp;├── \_\_init\_\_.py  
&emsp;&emsp;├── ETSNET  
&emsp;&emsp;&emsp;├── \_\_init\_\_.py  
&emsp;&emsp;&emsp;├── base.py  
&emsp;&emsp;&emsp;├── dice_loss.py  
&emsp;&emsp;&emsp;├── etsnet.py  
&emsp;&emsp;&emsp;├── fpn.py  
&emsp;&emsp;&emsp;├── resnet50.py  
&emsp;&emsp;&emsp;└── pse  
&emsp;&emsp;├── config.py  
&emsp;&emsp;├── dataset.py  
&emsp;&emsp;└── network_define.py  
&emsp;├── generate_hccn_file.py  
&emsp;├── test.py  
&emsp;└── train.py  
## Runing the example
### Train
#### Usage
download imagenet pretrained model from here.  
```
sh ./script/run_distribute_train.sh [PRETRAINED_MODEL]  
```
#### Result
epoch: 1 step: 20 , loss is 0.817393758893013  
epoch: 1 step: 40 , loss is 0.7769668325781822  
epoch: 1 step: 60 , loss is 0.7665704677502314  
...  
epoch: 608 step: 230 , loss is 0.33475895830146646  
epoch: 608 step: 250 , loss is 0.3369892504662275  
### Test
#### Usage
```
python test.py [CHECKPOINT_PATH]  
```
#### Result
get_data: 26.862831058387528ms, model_run: 190.04072216087448ms, post_process: 985.2245417768826ms: : 500it [11:31,  1.38s/it]  
### Eval Script for ICDAR2015
#### Usage
+ step 1: download eval method from [here](https://rrc.cvc.uab.es/?ch=4&com=tasks#TextLocalization).  
+ step 2: it is recommended to symlink the eval method root to $MINDSPORE/model_zoo/psenet/eval_ic15/. if your folder structure is different,you may need to change the corresponding paths in eval script files.  
```
sh ./script/eval_ic15.sh  
```
#### Result
Calculated!{"precision": 0.814796668299853, "recall": 0.8006740491092923, "hmean": 0.8076736279747451, "AP": 0}  