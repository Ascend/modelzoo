# DenseNet-121 for Tensorflow 

This repository provides a script and recipe to train the DenseNet-121 model .

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)  
  * [Default configuration](#default-configuration)
* [Data augmentation](#data-augmentation)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Command line arguments](#command-line-arguments)
  * [Training process](#training-process)
* [Performance](#performance)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
    * [Training performance results](#training-performance-results)

​    

## Model overview

DenseNet-121 is a classic image classification network that uses a Dense Block structure that is connected at each layer.To improve the model efficiency and reduce parameters, the BN-ReLU-Conv(1*1)-BN-ReLU-Conv(3*3)bottleneck layer is used and the 1*1 conv is used to limit the number of input channels at each layer in the Dense Block to 4k(k indicates the number of output channels at each layer).DenseNet can effectively mitigate gradient loss,facilitates feature transmission and reuse.
reference paper: Gao Huang,Zhuang Liu,Laurens van der Maaten,Kilian Q.Weinberger."Densely Connected Convolutional Networks."arXiv:1608.06993
 

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd  ModelZoo_AlexNet_TF_HARD
```

### 2. Download and preprocess the dataset
The model is compatible with the datasets on tensorflow official website.
1. download the ImageNet dataset
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. All images within one folder have the same label.

 **check json** 
Modify the *.JSON configuration file in the scripts directory to configure the relevant hardware IP information.
8P rank_table json Sample configuration file.
```
{
 "group_count": "1",
 "group_list": [
  {
   "group_name": "worker",
   "device_count": "8",
   "instance_count": "1",
   "instance_list": [
    {
     "devices":[
      {"device_id":"0","device_ip":"192.168.100.101"},
      {"device_id":"1","device_ip":"192.168.101.101"},
      {"device_id":"2","device_ip":"192.168.102.101"},
      {"device_id":"3","device_ip":"192.168.103.101"},
      {"device_id":"4","device_ip":"192.168.100.100"},
      {"device_id":"5","device_ip":"192.168.101.100"},
      {"device_id":"6","device_ip":"192.168.102.100"},
      {"device_id":"7","device_ip":"192.168.103.100"}
     ],
     "pod_name":"ascend8p",
     "server_id":"127.0.0.1"
    }
   ]
  }
 ],
 "status": "completed"
}
```
### 3. Train

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

 **Single card training process** 
Configure training parameters
First, in the script scritps/train_1p.sh,the training data set path and other parameters are configured.
```
python3.7 ${dname}/train.py --rank_size=1 \
    --mode=train \
    --max_train_steps=100 \
    --iterations_per_loop=10 \
    --data_dir=/opt/npu/slimImagenet \
    --display_every=10 \
    --log_dir=./model_1p \
    --log_name=densenet121_1p.log
```
Start training.
Starter card training(the script is scripts/run_1p.sh）:
```
bash run_1p.sh
```
 **8-card training process** 
Configure training parameters
First, in the script scritps/train_8p.sh,the training data set path and other parameters are configured.
```
python3.7 ${dname}/train.py --rank_size=8 \
    --mode=train_and_evaluate \
    --max_epochs=150 \
    --iterations_per_loop=5004 \
    --epochs_between_evals=5 \
    --data_dir=/opt/npu/slimImagenet \
    --lr=0.1 \
    --log_dir=./model_8p \
    --log_name=densenet121_8p.log
```
Start training.
Starter 8-card training(the script is scripts/run_8p.sh）:
```
bash run_8p.sh
```


### 4. Test
When testing, you need to modify the script startup parameters(script in scripts/test.sh).Configure mode to evaluate and set it in eval_dir configure the path of checkpoint file.
```
python3.7 ${dname}/train.py --rank_size=1 \
    --mode=evaluate \
    --data_dir=/opt/npu/slimImagenet \
    --eval_dir=${dname}/scripts/result/8p/0/model_8p \
    --log_dir=./ \
    --log_name=eval_densenet121.log > eval.log
```
Test instruction(script in scripts/test.sh)
```
bash test.sh
```

## Advanced
### Command-line options

```
  --data_dir                        train data dir
  --num_classes                     num of classes in ImageNet（default:1000)
  --image_size                      image size of the dataset
  --batch_size                      mini-batch size (default: 128) per npu
  --pretrained                      path of pretrained model
  --lr                              initial learning rate(default: 0.06)
  --max_epochs                      max number of epoch to train the model(default: 150)
  --warmup_epochs                   warmup epoch(when batchsize is large)
  --weight_decay                    weight decay (default: 1e-4)
  --momentum                        momentum(default: 0.9)
  --label_smoothing                 use label smooth in CE, (default 0.1)
  --save_summary_steps              logging interval(dafault:100)
  --log_dir                         path to save checkpoint and log
  --log_name                        name of log file
  --save_checkpoints_steps          the interval to save checkpoint
  --mode                            mode to run the program (train, evaluate)
  --checkpoint_dir                  path to checkpoint for evaluation
  --max_train_steps                 max number of training steps 
  --synthetic                       whether to use synthetic data or not
  --version                         weight initialization for model
  --do_checkpoint                   whether to save checkpoint or not 
  --rank_size                       local rank of distributed(default: 0)
  --group_size                      world size of distributed(default: 1)
  --max_train_steps                 number of training step , default : None, when set ,it will override the max_epoch
```
for a complete list of options, please refer to `train.py`

### Training process

Start single card or multi card training through the training instructions in "quick start". Single card and multi card support single card and eight card network training by running different scripts. Different experimental parameters are supported. For specific process, refer to the example of "quick start".

The storage path of the model is "results/", including the training log and checkpoints files. Take 8 cards, OS = 16 training as an example, the loss information is in the file results/s16/r1/0/resnet_101/training.log An example is as follows.
```
step:   500  epoch:  3.0  FPS:  148.4  loss:0.106533  reg_loss:0.3788784  total_loss:0.485411  lr:0.02792
step:  1000  epoch:  6.0  FPS:  268.8  loss:0.456593  reg_loss:0.3740377  total_loss:0.830631  lr:0.02769
step:  1500  epoch:  9.1  FPS:  270.0  loss:0.184617  reg_loss:0.3686840  total_loss:0.553301  lr:0.02732
step:  2000  epoch: 12.1  FPS:  271.3  loss:0.120532  reg_loss:0.3624873  total_loss:0.483020  lr:0.02679
step:  2500  epoch: 15.1  FPS:  270.2  loss:0.197548  reg_loss:0.3565496  total_loss:0.554098  lr:0.02613
step:  3000  epoch: 18.1  FPS:  267.0  loss:0.166731  reg_loss:0.3508141  total_loss:0.517545  lr:0.02533
step:  3500  epoch: 21.2  FPS:  271.1  loss:0.085695  reg_loss:0.3450070  total_loss:0.430702  lr:0.02441
step:  4000  epoch: 24.2  FPS:  271.8  loss:0.185468  reg_loss:0.3397226  total_loss:0.525190  lr:0.02337
step:  4500  epoch: 27.2  FPS:  273.1  loss:0.092546  reg_loss:0.3341669  total_loss:0.426712  lr:0.02223
step:  5000  epoch: 30.2  FPS:  273.2  loss:0.106140  reg_loss:0.3286971  total_loss:0.434837  lr:0.02100
```

### Verification / reasoning process

When OS = 16, checkpoint=resnet_v1_101_beta training is completed, the test script test_1p_s16_beta.sh is executed according to the test process in "quick start".
For the OS = 8 experiment, after the experiment training, there are three groups of different test settings, directly run the corresponding script.
```
bash scripts/test_1p_s16_beta.sh
```
The script will automatically execute the verification process, and the verification results will be output to the eval/0/resnet/training.log file.
```
eval/miou_1.0_class_0 :  0.94257
eval/miou_1.0_class_1 :  0.88760
eval/miou_1.0_class_10 :  0.86478
eval/miou_1.0_class_11 :  0.52987
eval/miou_1.0_class_12 :  0.88413
eval/miou_1.0_class_13 :  0.86907
eval/miou_1.0_class_14 :  0.87423
eval/miou_1.0_class_15 :  0.85519
eval/miou_1.0_class_16 :  0.62811
eval/miou_1.0_class_17 :  0.85584
eval/miou_1.0_class_18 :  0.49328
eval/miou_1.0_class_19 :  0.83555
eval/miou_1.0_class_2 :  0.53385
eval/miou_1.0_class_20 :  0.73652
eval/miou_1.0_class_3 :  0.87449
eval/miou_1.0_class_4 :  0.64409
eval/miou_1.0_class_5 :  0.77315
eval/miou_1.0_class_6 :  0.93965
eval/miou_1.0_class_7 :  0.88311
eval/miou_1.0_class_8 :  0.92532
eval/miou_1.0_class_9 :  0.42902
loss :  0.28353
global_step :  15000.00000
checkpoint: /home/models/deeplabv3/results/s16/r2/0/resnet_101/model.ckpt-15000
mean IOU is : 0.7742580952380952
```










