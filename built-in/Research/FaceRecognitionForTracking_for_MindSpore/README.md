# Contents

- [Face Recognition For Tracking Description](#face-recognition-for-tracking-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)  
- [ModelZoo Homepage](#modelzoo-homepage)


# [Face Recognition For Tracking Description](#contents)

This is a face recognition for tracking network based on Resnet, with support for training and evaluation on Ascend910.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"


# [Model Architecture](#contents)

Face Recognition For Tracking uses a Resnet network for performing feature extraction.


# [Dataset](#contents)

We use about 10K face images as training dataset and 2K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. Labeled Faces in the Wild)
The directory structure is as follows:
```
.
└─ dataset
  ├─ train dataset
    ├─ ID1
      ├─ ID1_0001.jpg
      ├─ ID1_0002.jpg
      ...
    ├─ ID2
      ...
    ├─ ID3
      ...
    ...
  ├─ test dataset
    ├─ ID1
      ├─ ID1_0001.jpg
      ├─ ID1_0002.jpg
      ...
    ├─ ID2
      ...
    ├─ ID3
      ...
    ...
```

# [Environment Requirements](#contents)

- Hardware(Ascend)
  - Prepare hardware environment with Ascend processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below:
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)


# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:
```
.
└─ Face Recognition For Tracking
  ├─ README.md
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    └─ run_export.sh                        # launch exporting air model
  ├─ src
    ├─ config.py                            # parameter configuration
    ├─ dataset.py                           # dataset loading and preprocessing for training
    ├─ reid.py                              # network backbone
    ├─ reid_for_export.py                   # network backbone for export
    ├─ log.py                               # log function
    ├─ loss.py                              # loss function
    ├─ lr_generator.py                      # generate learning rate
    └─ me_init.py                           # network initialization
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  ├─ export.py                              # export air model
  └─ generate_rank_table.py                 # generate rank table
```


## [Running Example](#contents)

### Prepare

- Generate ranktable on Ascend910:(If you want to use your own ranktable, skip this step)
    ```
    python generate_rank_table.py --server_id=[SERVER_ID] --nproc_per_node=8
    ```

### Train

- Stand alone mode

    ```
    cd ./scripts
    sh run_standalone_train.sh [DATA_DIR] [USE_DEVICE_ID]
    ```
    or (fine-tune)
    ```
    cd ./scripts
    sh run_standalone_train.sh [DATA_DIR] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```
    for example:
    ```
    cd ./scripts
    sh run_standalone_train.sh /home/train_dataset 0 /home/a.ckpt
    ```

- Distribute mode (recommended)

    ```
    cd ./scripts
    sh run_distribute_train.sh [DATA_DIR] [RANK_TABLE]
    ```
    or (fine-tune)
    ```
    cd ./scripts
    sh run_distribute_train.sh [DATA_DIR] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```
    for example:
    ```
    cd ./scripts
    sh run_distribute_train.sh /home/train_dataset ./rank_table_8p.json /home/a.ckpt
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

```
epoch[0], iter[10], loss:43.314265, 8574.83 imgs/sec, lr=0.800000011920929
epoch[0], iter[20], loss:45.121095, 8915.66 imgs/sec, lr=0.800000011920929
epoch[0], iter[30], loss:42.342847, 9162.85 imgs/sec, lr=0.800000011920929
epoch[0], iter[40], loss:39.456583, 9178.83 imgs/sec, lr=0.800000011920929

...
epoch[179], iter[14900], loss:1.651353, 13001.25 imgs/sec, lr=0.02500000037252903
epoch[179], iter[14910], loss:1.532123, 12669.85 imgs/sec, lr=0.02500000037252903
epoch[179], iter[14920], loss:1.760322, 13457.81 imgs/sec, lr=0.02500000037252903
epoch[179], iter[14930], loss:1.694281, 13417.38 imgs/sec, lr=0.02500000037252903
```

### Evaluation

```
cd ./scripts
sh run_eval.sh [EVAL_DIR] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```
for example:
```
cd ./scripts
sh run_eval.sh /home/test_dataset 0 /home/a.ckpt
```

You will get the result as following in "./scripts/device0/eval.log" or txt file in [PRETRAINED_BACKBONE]'s folder:

```
0.5: 0.9273788254649683@0.020893691253149882 
0.3: 0.8393850978779193@0.07438552515516506 
0.1: 0.6220871197028316@0.1523084478903911 
0.01: 0.2683641598437038@0.26217882879427634 
0.001: 0.11060269148211463@0.34509718987101223 
0.0001: 0.05381678898728808@0.4187797093636618 
1e-05: 0.035770748447963394@0.5053771466191392 
```

### Convert model

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```
cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```


# [Model Description](#contents)
## [Performance](#contents)

### Training Performance 

| Parameters                 | Face Recognition For Tracking                               |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 09/30/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | 10K images                                                  |
| Training Parameters        | epoch=180, batch_size=16, momentum=0.9                      |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Speed                      | 1pc: 8~10 ms/step; 8pcs: 9~11 ms/step                       |
| Total time                 | 1pc: 1 hours; 8pcs: 0.1 hours                               |
| Checkpoint for Fine tuning | 17M (.ckpt file)                                            |


### Evaluation Performance

| Parameters          |Face Recognition For Tracking|
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/30/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | 2K images                   |
| batch_size          | 128                         |
| outputs             | recall                      |
| Recall(8pcs)        | 0.62(FAR=0.1)               |
| Model for inference | 17M (.ckpt file)            |


# [ModelZoo Homepage](#contents)
Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).