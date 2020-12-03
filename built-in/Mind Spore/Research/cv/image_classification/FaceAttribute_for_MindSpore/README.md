# Contents

- [Face Attribute Description](#face-attribute-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)  
- [ModelZoo Homepage](#modelzoo-homepage)


# [Face Attribute Description](#contents)

This is a Face Attributes Recognition network based on Resnet18, with support for training and evaluation on Ascend910.

ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

[Paper](https://arxiv.org/pdf/1512.03385.pdf):  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"


# [Model Architecture](#contents)

Face Attribute uses a modified-Resnet18 network for performing feature extraction.


# [Dataset](#contents)

This network can recognize the age/gender/mask from a human face. The default rule is:
```
age:
    0: 0~2 years
    1: 3~9 years
    2: 10~19 years
    3: 20~29 years
    4: 30~39 years
    5: 40~49 years
    6: 50~59 years
    7: 60~69 years
    8: 70+ years

gender:
    0: male
    1: female

mask:
    0: wearing mask
    1: without mask
```

We use about 91K face images as training dataset and 11K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. FairFace and RWMFD)
    
- step 1: The dataset should be saved in a txt file, which contain the following contents:
    ```
    [PATH_TO_IMAGE]/1.jpg [LABEL_AGE] [LABEL_GENDER] [LABEL_MASK]
    [PATH_TO_IMAGE]/2.jpg [LABEL_AGE] [LABEL_GENDER] [LABEL_MASK]
    [PATH_TO_IMAGE]/3.jpg [LABEL_AGE] [LABEL_GENDER] [LABEL_MASK]
    ...
    ```
    
    The value range of [LABEL_AGE] is [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], -1 means the label should be ignored.
    
    The value range of [LABEL_GENDER] is [-1, 0, 1], -1 means the label should be ignored.
    
    The value range of [LABEL_MASK] is [-1, 0, 1], -1 means the label should be ignored.

- step 2: Convert the dataset to mindrecord:
    ```
    python data_to_mindrecord_train.py
    ```
    or
    ```
    python data_to_mindrecord_eval.py
    ```
    If your dataset is too big to convert at a time, you can add data to an existed mindrecord in turn:
    ```
    python data_to_mindrecord_train_append.py
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
└─ Face Attribute
  ├─ README.md
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    └─ run_export.sh                        # launch exporting air model
  ├─ src
    ├─ FaceAttribute
      ├─ cross_entropy.py                   # cross entroy loss
      ├─ custom_net.py                      # network unit
      ├─ loss_factory.py                    # loss function
      ├─ head_factory.py                    # network head
      ├─ resnet18.py                        # network backbone
      ├─ head_factory_softmax.py            # network head with softmax
      └─ resnet18_softmax.py                # network backbone with softmax
    ├─ config.py                            # parameter configuration
    ├─ dataset_eval.py                      # dataset loading and preprocessing for evaluating
    ├─ dataset_train.py                     # dataset loading and preprocessing for training
    ├─ logging.py                           # log function
    └─ lrsche_factory.py                    # generate learning rate
  ├─ train.py                               # training scripts
  ├─ eval.py                                # evaluation scripts
  ├─ export.py                              # export air model
  ├─ generate_rank_table.py                 # generate rank table
  ├─ data_to_mindrecord_train.py            # convert dataset to mindrecord for training
  ├─ data_to_mindrecord_train_append.py     # add dataset to an existed mindrecord for training
  └─ data_to_mindrecord_eval.py             # convert dataset to mindrecord for evaluating
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
    sh run_standalone_train.sh [MINDRECORD_FILE] [USE_DEVICE_ID]
    ```
    or (fine-tune)
    ```
    cd ./scripts
    sh run_standalone_train.sh [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
    ```
    for example:
    ```
    cd ./scripts
    sh run_standalone_train.sh /home/train.mindrecord 0 /home/a.ckpt
    ```

- Distribute mode (recommended)

    ```
    cd ./scripts
    sh run_distribute_train.sh [MINDRECORD_FILE] [RANK_TABLE]
    ```
    or (fine-tune)
    ```
    cd ./scripts
    sh run_distribute_train.sh [MINDRECORD_FILE] [RANK_TABLE] [PRETRAINED_BACKBONE]
    ```
    for example:
    ```
    cd ./scripts
    sh run_distribute_train.sh /home/train.mindrecord ./rank_table_8p.json /home/a.ckpt
    ```

You will get the loss value of each step as following in "./output/[TIME]/[TIME].log" or "./scripts/device0/train.log":

```
epoch[0], iter[0], loss:4.489518, 12.92 imgs/sec
epoch[0], iter[10], loss:3.619693, 13792.76 imgs/sec
epoch[0], iter[20], loss:3.580932, 13817.78 imgs/sec
epoch[0], iter[30], loss:3.574254, 7834.65 imgs/sec
epoch[0], iter[40], loss:3.557742, 7884.87 imgs/sec

...
epoch[69], iter[6120], loss:1.225308, 9561.00 imgs/sec
epoch[69], iter[6130], loss:1.209557, 8913.28 imgs/sec
epoch[69], iter[6140], loss:1.158641, 9755.81 imgs/sec
epoch[69], iter[6150], loss:1.167064, 9300.77 imgs/sec
```

### Evaluation

```
cd ./scripts
sh run_eval.sh [MINDRECORD_FILE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```
for example:
```
cd ./scripts
sh run_eval.sh /home/eval.mindrecord 0 /home/a.ckpt
```

You will get the result as following in "./scripts/device0/eval.log" or txt file in [PRETRAINED_BACKBONE]'s folder:

```
age accuracy:  0.45773233522001094
gen accuracy:  0.8950155194449516
mask accuracy:  0.992539346357495
gen precision:  0.8869598765432098
gen recall:  0.8907400232468036
gen f1:  0.88884593079451
mask precision:  1.0
mask recall:  0.998539346357495
mask f1:  0.9992691394116572
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

| Parameters                 | Face Attribute                                              |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 09/30/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | 91K images                                                  |
| Training Parameters        | epoch=70, batch_size=128, momentum=0.9, lr=0.001            |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy                                       |
| outputs                    | probability                                                 |
| Speed                      | 1pc: 200~250 ms/step; 8pcs: 100~150 ms/step                 |
| Total time                 | 1pc: 2.5 hours; 8pcs: 0.3 hours                             |
| Checkpoint for Fine tuning | 88M (.ckpt file)                                            |


### Evaluation Performance

| Parameters          | Face Attribute              |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/30/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | 11K images                  |
| batch_size          | 1                           |
| outputs             | accuracy                    |
| Accuracy(8pcs)      | age:45.7%                   |
|                     | gender:89.5%                |
|                     | mask:99.2%                  |
| Model for inference | 88M (.ckpt file)            |


# [ModelZoo Homepage](#contents)
Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).