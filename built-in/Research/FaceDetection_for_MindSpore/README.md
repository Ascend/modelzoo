# Contents

- [Face Detection Description](#face-detection-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)  
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Running Example](#running-example)
- [Model Description](#model-description)
    - [Performance](#performance)  
- [ModelZoo Homepage](#modelzoo-homepage)


# [Face Detection Description](#contents)

This is a Face Detection network based on Yolov3, with support for training and evaluation on Ascend910.

You only look once (YOLO) is a state-of-the-art, real-time object detection system. YOLOv3 is extremely fast and accurate. 

Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.
 YOLOv3 use a totally different approach. It apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities. 

[Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf):  YOLOv3: An Incremental Improvement. Joseph Redmon, Ali Farhadi,
University of Washington


# [Model Architecture](#contents)

Face Detection uses a modified-DarkNet53 network for performing feature extraction. It has 45 convolutional layers.


# [Dataset](#contents)

We use about 13K images as training dataset and 3K as evaluating dataset in this example, and you can also use your own datasets or open source datasets (e.g. WiderFace)
    
- step 1: The dataset should follow the Pascal VOC data format for object detection. The directory structure is as follows:(Because of the small input shape of network, we remove the face lower than 50*50 at 1080P in evaluating dataset )
    ```
        .
        └─ dataset
          ├─ Annotations
            ├─ img1.xml
            ├─ img2.xml
            ├─ ...
          ├─ JPEGImages
            ├─ img1.jpg
            ├─ img2.jpg
            ├─ ...
          └─ ImageSets
            └─ Main
              └─ train.txt or test.txt
    ```

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

- Hardware（Ascend）
  - Prepare hardware environment with Ascend processor. If you want to try Ascend  , please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources. 
- Framework
  - [MindSpore](http://10.90.67.50/mindspore/archive/20200506/OpenSource/me_vm_x86/)
- For more information, please check the resources below：
  - [MindSpore tutorials](https://www.mindspore.cn/tutorial/zh-CN/master/index.html) 
  - [MindSpore API](https://www.mindspore.cn/api/zh-CN/master/index.html)


# [Script Description](#contents)

## [Script and Sample Code](#contents)

The entire code structure is as following:
```
.
└─ Face Detection
  ├─ README.md
  ├─ scripts
    ├─ run_standalone_train.sh              # launch standalone training(1p) in ascend
    ├─ run_distribute_train.sh              # launch distributed training(8p) in ascend
    ├─ run_eval.sh                          # launch evaluating in ascend
    └─ run_export.sh                        # launch exporting air model
  ├─ src
    ├─ FaceDetection
      ├─ nms                                # NMS unit
        ├─ cpu_nms.c
        ├─ cpu_nms.pyx
        └─ py_cpu_nms.py
      ├─ Makefile                           # NMS unit makefile
      ├─ setup.py                           # NMS setup file
      ├─ voc_wrapper.py                     # get detection results
      ├─ nms_wrapper.py                     # NMS
      ├─ yolo_loss.py                       # loss function
      ├─ yolo_postprocess.py                # post process
      └─ yolov3.py                          # network
    ├─ config.py                            # parameter configuration
    ├─ data_preprocess.py                   # preprocess
    ├─ logging.py                           # log function
    ├─ lrsche_factory.py                    # generate learning rate
    ├─ network_define.py                    # network define
    └─ transforms.py                        # data transforms
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
  
- Compile NMS module:
    ```
    cd ./src/FaceDetection
    make -j
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
rank[0], iter[0], loss[318555.8], overflow:False, loss_scale:1024.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
rank[0], iter[1], loss[95394.28], overflow:True, loss_scale:1024.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
rank[0], iter[2], loss[81332.92], overflow:True, loss_scale:512.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
rank[0], iter[3], loss[27250.805], overflow:True, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
...

rank[0], iter[62496], loss[2218.6282], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
rank[0], iter[62497], loss[3788.5146], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
rank[0], iter[62498], loss[3427.5479], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
rank[0], iter[62499], loss[4294.194], overflow:False, loss_scale:256.0, lr:6.24999984211172e-06, batch_images:(64, 3, 448, 768), batch_labels:(64, 200, 6)
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

You will get the result as following in "./scripts/device0/eval.log":

```
calculate [recall | persicion | ap]...
Saving ../../results/0-2441_61000/.._.._results_0-2441_61000_face_AP_0.760.png
```

And the detect result and P-R graph will also be saved in "./results/[MODEL_NAME]/"

### Convert model

If you want to infer the network on Ascend 310, you should convert the model to AIR:

```
cd ./scripts
sh run_export.sh [BATCH_SIZE] [USE_DEVICE_ID] [PRETRAINED_BACKBONE]
```


# [Model Description](#contents)
## [Performance](#contents)

### Training Performance 

| Parameters                 | Face Detection                                              |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 09/30/2020 (month/day/year)                                 |
| MindSpore Version          | 1.0.0                                                       |
| Dataset                    | 13K images                                                  |
| Training Parameters        | epoch=2500, batch_size=64, momentum=0.5                     |
| Optimizer                  | Momentum                                                    |
| Loss Function              | Softmax Cross Entropy, Sigmoid Cross Entropy, SmoothL1Loss  |
| outputs                    | boxes and label                                             |
| Speed                      | 1pc: 800~850 ms/step; 8pcs: 1000~1150 ms/step               |
| Total time                 | 1pc: 120 hours; 8pcs: 18 hours                              |
| Checkpoint for Fine tuning | 37M (.ckpt file)                                            |


### Evaluation Performance

| Parameters          | Face Detection              |
| ------------------- | --------------------------- |
| Model Version       | V1                          |
| Resource            | Ascend 910                  |
| Uploaded Date       | 09/30/2020 (month/day/year) |
| MindSpore Version   | 1.0.0                       |
| Dataset             | 3K images                   |
| batch_size          | 1                           |
| outputs             | mAP                         |
| Accuracy            | 8pcs: 76.0%                 |
| Model for inference | 37M (.ckpt file)            |


# [ModelZoo Homepage](#contents)
Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).