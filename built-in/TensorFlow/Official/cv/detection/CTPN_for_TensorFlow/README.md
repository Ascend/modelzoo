# CTPN for Tensorflow 

This repository provides a script and recipe to train the CTPN model. The code is based on https://github.com/eragonruan/text-detection-ctpn,
modifications are made to run on NPU. Original README file can be found in `README_ORI.md`  

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

## Model overview

CTPN model from
`Zhi Tian, Weilin Huang, Tong He, Pan He, Yu Qiao "Detecting Text in Natural Image with Connectionist Text Proposal Network and Its Application to Scene Text Recognition". <https://arxiv.org/abs/1609.03605>.`

reference implementation:  <https://github.com/eragonruan/text-detection-ctpn>

### Model architecture



### Default configuration

The following sections introduce the default configurations and hyperparameters for CTPN  model. We reproduce training setups 
on mlt datasets, evaluate on ICDAR2013 test. See [Results](#results) for details.

For detailed hpyerparameters, please refer to corresponding scripts under directory `main/train_npu.py`
#### Optimizer

This model uses Adam optimizer from Tensorflow with the following hyperparameters:

- LR schedule: piecewise_constant

#### Data augmentation

This model uses the following data augmentation:

- For training:
  - resize to (H,W)=(600,900)
- For inference:
  - resize to (H,W)=(600,900)


## Setup
The following section lists the requirements to start training the CTPN model.
### Requirements

see `requirements.txt`

### nms and bbox
nms and bbox utils are written in cython, hence you have to build the library first.
```shell
cd utils/bbox
chmod +x make.sh
./make.sh
```
It will generate a nms.so and a bbox.so in current folder.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd  CTPN_for_TensorFlow/
```

### 2. Download the dataset and Pretrain model

- First, download the pre-trained model of VGG net and put it in data/vgg_16.ckpt. you can download it from [tensorflow/models](https://github.com/tensorflow/models/tree/1af55e018eebce03fb61bba9959a04672536107d/research/slim)
- Second, download the dataset we prepared from [google drive](https://drive.google.com/file/d/1npxA_pcEvIa4c42rho1HgnfJ7tamThSy/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1nbbCZwlHdgAI20_P9uw9LQ). put the downloaded data in data/dataset/mlt, then start the training.
- Also, you can prepare your own dataset according to the following steps. 
- Modify the DATA_FOLDER and OUTPUT in utils/prepare/split_label.py according to your dataset. And run split_label.py in the root
```shell
python ./utils/prepare/split_label.py
```


### 3. Train

Simplely run:
   
- For training on single NPU device, execute the shell script `run_1p.sh`, e.g.
  ```
   bash run_1p.sh
  ```

### 4. Test
Three datases are used to evaluate the trained model. To test, just run test script `bash eval.sh ${DIR_TO_CHECKPOINTS}` (replace the ${DIR_TO_CHECKPOINTS}  with your own path to checkpoint file). 

After finished, test results will be shown on the terminal:
  ```
   bash eval.sh ${DIR_TO_CHECKPOINTS}
   ...
   {"precision": xxx, "recall": xxx, "hmean": xxx, "AP": 0}
  ```


## Advanced
### Commmand-line options


```
  --learning_rate                 1e-5, default ./
  --max_steps                     50000, default data/
  --decay_steps	                  30000, default
  --pretrained_model_path         pretrain model path, default: data/vgg_16.ckpt                    
```
for a complete list of options, please refer to `main/train_npu.py`

### Training process

All the results of the training will be stored in the directory `checkpoint`.
Script will store:
 - checkpoints
 - events
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.


#### Evaluation results 
The accuracy is measured on ICDAR2013 test setup by run the shell script:
```
bash eval.sh
```
and get precision,recall,Hmean as following:
| **Precision** | **Recall**       | **Hmean** |
| :------: | :---------------: |:---------------:  |
|    0.846     | 0.737              |  0.788     |

#### Training performance 

| **NPUs** | batch size        | train performance |
| :------: | :---------------: |:---------------:  |
|    1     | 64*8              |       |