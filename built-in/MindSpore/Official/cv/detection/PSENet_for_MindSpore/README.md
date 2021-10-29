# Contents

- [Contents](#contents)
- [PSENet Description](#psenet-description)
- [PSENet Example](#psenet-example)
    - [Description](#description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [run test code](#run-test-code)
        - [Eval Script for ICDAR2015](#eval-script-for-icdar2015)
            - [Usage](#usage)
            - [Result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#inference-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)

# [PSENet Description](#contents)
With the development of convolutional neural network, scene text detection technology has been developed rapidly. However, there are still two problems in this algorithm, which hinders its application in industry. On the one hand, most of the existing algorithms require quadrilateral bounding boxes to accurately locate arbitrary shape text. On the other hand, two adjacent instances of text can cause error detection overwriting both instances. Traditionally, a segmentation-based approach can solve the first problem, but usually not the second. To solve these two problems, a new PSENet (PSENet) is proposed, which can accurately detect arbitrary shape text instances. More specifically, PSENet generates different scale kernels for each text instance and gradually expands the minimum scale kernel to a text instance with full shape. Because of the large geometric margins between the minimum scale kernels, our method can effectively segment closed text instances, making it easier to detect arbitrary shape text instances. The effectiveness of PSENet has been verified by numerous experiments on CTW1500, full text, ICDAR 2015, and ICDAR 2017 MLT. 

[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html):  Wenhai Wang, Enze Xie, Xiang Li, Wenbo Hou, Tong Lu, Gang Yu, Shuai Shao; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9336-9345


# PSENet Example
## Description
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

# [Dataset](#contents)
Dataset supported: `ICDAR2015`
A training set of 1000 images containing about 4500 readable words
A testing set containing about 2000 readable words

# [Environment Requirements](#contents)
- Hardware（Ascend）
  - Prepare hardware environment with Ascend processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](http://www.mindspore.cn/install/en)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)
- install Mindspore
- install [pyblind11](https://github.com/pybind/pybind11)
- install [Opencv3.4](https://docs.opencv.org/3.4.9/d7/d9f/tutorial_linux_install.html)

# [Quick Start](#contents)
After installing MindSpore via the official website, you can start training and evaluation as follows: 
```python
# run distributed training example
sh scripts/run_distribute_train.sh pretrained_model.ckpt

#download opencv library
download pyblind11, opencv3.4

#install pyblind11 opencv3.4
setup pyblind11(install the library by the pip command)
setup opencv3.4(compile source code install the library)

#enter the path ,run Makefile to product file
cd ./src/ETSNET/pse/;make

#run test.py
python test.py --ckpt=pretrained_model.ckpt

# download eval method from websit of ICDAR2015
# click "My Methods" button,then download Evaluation Scripts
# download script.py
# run evaluation example
sh scripts/run_eval_ascend.sh
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)
```
└── PSENet  
	├── README.md                           // descriptions about PSENet
	├── scripts  
		├── run_distribute_train.sh  		// shell script for distributed
		└── run_eval_ascend.sh  			// shell script for evaluation 
	├── src  
		├── __init__.py  
        ├── generate_hccn_file.py           // creating rank.json
		├── ETSNET  
			├── __init__.py  
			├── base.py                     // convolution and BN operator
			├── dice_loss.py                // calculate PSENet loss value
			├── etsnet.py                   // Subnet in  PSENet
			├── fpn.py                      // Subnet in  PSENet
			├── resnet50.py                 // Subnet in  PSENet
			├── pse                         // Subnet in  PSENet
                ├── __init__.py
                ├── adaptor.cpp
                ├── adaptor.h
                ├── Makefile
		├── config.py                       // parameter configuration 
		├── dataset.py                      // creating dataset
		└── network_define.py               // PSENet architecture
	├── test.py                             // test script 
	└── train.py                            // training script

```

## [Script Parameters](#contents)

```python
Major parameters in train.py and config.py are:

--pre_trained: Whether training from scratch or training based on the
               pre-trained model.Optional values are True, False. 
--device_id: Device ID used to train or evaluate the dataset. Ignore it
             when you use train.sh for distributed training.
--device_num: devices used when you use train.sh for distributed training.

```


## [Training Process](#contents)

### Distributed Training
```
sh scripts/run_distribute_train.sh pretrained_model.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file 
`device[X]/test_*.log`. The loss value will be achieved as follows:

```
# grep "epoch: " device_*/loss.log
device_0/log:epoch: 1, step: 20, loss is 0.80383
device_0/log:epcoh: 2, step: 40, loss is 0.77951
...
device_1/log:epoch: 1, step: 20, loss is 0.78026
device_1/log:epcoh: 2, step: 40, loss is 0.76629

```

## [Evaluation Process](#contents)
### run test code
python test.py --ckpt=./device*/ckpt*/ETSNet-*.ckpt

### Eval Script for ICDAR2015
#### Usage
+ step 1: download eval method from website of ICDAR2015 of rcc.
+ step 2: click "My Methods" button,then download Evaluation Scripts.
+ step 3: it is recommended to symlink the eval method root to $MINDSPORE/model_zoo/psenet/eval_ic15/. if your folder structure is different,you may need to change the corresponding paths in eval script files.  
```
sh ./script/run_eval_ascend.sh.sh  
```
#### Result
Calculated!{"precision": 0.814796668299853, "recall": 0.8006740491092923, "hmean": 0.8076736279747451, "AP": 0}


# [Model Description](#contents)
## [Performance](#contents)

### Evaluation Performance 

| Parameters                 | PSENet                                                   |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | Inception V1                                                |
| Resource                   | Ascend 910 ；CPU 2.60GHz，192cores；Memory，755G             |
| uploaded Date              | 09/15/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0-alpha                                                   |
| Dataset                    | ICDAR2015                                                   |
| Training Parameters        | start_lr=0.1; lr_scale=0.1                                  |
| Optimizer                  | SGD                                                         |
| Loss Function              | LossCallBack                                                |
| outputs                    | probability                                                 |
| Loss                       | 0.35                                                        |
| Speed                      | 1pc: 444 ms/step;  4pcs: 446 ms/step                        |
| Total time                 | 1pc: 75.48 h;  4pcs: 18.87 h                                |
| Parameters (M)             | 27.36                                                       |
| Checkpoint for Fine tuning | 109.44M (.ckpt file)                                        |
| Scripts                    | https://gitee.com/mindspore/mindspore/tree/master/model_zoo/psenet |


### Inference Performance

| Parameters          | PSENet                      |
| ------------------- | --------------------------- |
| Model Version       | Inception V1                |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/15/2020 (month/day/year) |
| MindSpore Version   | 1.0-alpha                   |
| Dataset             | ICDAR2015                   |
| outputs             | probability                 |
| Accuracy            | 1pc: 81%;  4pcs: 81%   |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html). Following the steps below, this is a simple example:

```
# Load unseen dataset for inference
dataset = dataset.create_dataset(cfg.data_path, 1, False)

# Define model 
config.INFERENCE = False
net = ETSNet(config)
net = net.set_train()
param_dict = load_checkpoint(args.pre_trained)
load_param_into_net(net, param_dict)
print('Load Pretrained parameters done!')

criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)

lrs = lr_generator(start_lr=1e-3, lr_scale=0.1, total_iters=config.TRAIN_TOTAL_ITER)
opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs, momentum=0.99, weight_decay=5e-4)

# warp model
net = WithLossCell(net, criterion)
net = TrainOneStepCell(net, opt)

time_cb = TimeMonitor(data_size=step_size)
loss_cb = LossCallBack(per_print_times=20)
# set and apply parameters of check point
ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=2)
ckpoint_cb = ModelCheckpoint(prefix="ETSNet", config=ckpoint_cf, directory=config.TRAIN_MODEL_SAVE_PATH)

model = Model(net)
model.train(config.TRAIN_REPEAT_NUM, ds, dataset_sink_mode=False, callbacks=[time_cb, loss_cb, ckpoint_cb])

# Load pre-trained model
param_dict = load_checkpoint(cfg.checkpoint_path)
load_param_into_net(net, param_dict)
net.set_train(False)

# Make predictions on the unseen dataset
acc = model.eval(dataset)
print("accuracy: ", acc)
```
