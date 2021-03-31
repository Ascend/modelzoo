# UNet3D for Tensorflow 

This repository provides scripts and recipe to train the UNet3D model to achieve state-of-the-art accuracy.

## Table Of Contents

- [UNet3D for Tensorflow](#unet3d-for-tensorflow)
  - [Table Of Contents](#table-of-contents)
  - [Model overview](#model-overview)
    - [Model architecture](#model-architecture)
    - [Default configuration](#default-configuration)
      - [Optimizer](#optimizer)
      - [Data augmentation](#data-augmentation)
  - [Setup](#setup)
    - [Requirements](#requirements)
  - [Quick Start Guide](#quick-start-guide)
    - [1. Clone the respository and change the working directory](#1-clone-the-respository-and-change-the-working-directory)
    - [2. Download and preprocess the dataset](#2-download-and-preprocess-the-dataset)
    - [3. Set environment](#3-set-environment)
    - [4. Train](#4-train)
    - [5. Test](#5-test)
  - [Advanced](#advanced)
    - [Command-line options](#command-line-options)
    - [Training process](#training-process)
  - [Performance](#performance)
    - [Result](#result)
      - [Training accuracy results](#training-accuracy-results)
      - [Training performance results](#training-performance-results)


    

## Model overview

In this repository, we implement a 3D-UNet introduced from paper [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650) with modifications described in [No New-Net](https://arxiv.org/pdf/1809.10483). Our code is based on NVIDIA's [implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical).

### Model architecture

The model architecture can be found from the reference paper.

### Default configuration

The following sections introduce the default configurations and hyperparameters for UNet3D model.

#### Optimizer

This model uses Adam optimizer from Tensorflow with the following hyperparameters (for 8 NPUs):

- Learning rate (LR) : 0.0002
- Batch size : 2*8
- We train for:
  - 16,000 (16,000 / number of NPUs) iterations for each fold
  
Users can find more detailed parameters from the source code.

#### Data augmentation

This model uses the following data augmentation:

- For training:
  - RandomCrop3D((128, 128, 128))
  - Cast to tf.float32
  - RandomHorizontalFlip with threshold of 0.5
  - Normalize
  - RandomBrightnessCorrection
- For inference:
  - CenterCrop((224, 224, 155))
  - Cast to tf.float32
  - Normalize
  - PadXYZ

For more details, we refer readers to read the corresponding source code.

## Setup
The following section lists the requirements to start training the UNet3D model.

### Requirements

Tensorflow 1.15.0

DLLogger 0.1.0

## Quick Start Guide

### 1. Clone the respository and change the working directory

```shell
git clone xxx
cd ModelZoo_UNet3D_TF_HARD/00-access
```

### 2. Download and preprocess the dataset

1. Register and download data from [Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/).
2. You can use the `dataset/preprocess_data.py` script to convert the raw data into tfrecord format used for training and evaluation with a command like
    ```python
    python dataset/preprocess_data.py -i path/ori_data -o path/tf_data -v
    ```

### 3. Set environment

Set environment variable like *LD_LIBRARY_PATH*, *PYTHONPATH* and *PATH* to match your system before training and testing. Configure json file for multiple NPUs.

### 4. Train
In this repository, we provide scripts to obtain training performance for single NPU, both training performance and training accuracy for 8 NPUs.
- training performance on a single NPU
    - **edit** *train_1p_performance.sh* (see example below)
    - bash run_1p_performance.sh
- training performance on 8 NPUs
    - **edit** *train_8p_performance.sh* (see example below)
    - bash run_8p_performance.sh
- training accuracy on 8 NPUs 
    - **edit** *train_8p_accuracy.sh* (see example below)
    - bash run_8p_accuracy.sh /npu/z00438116/BraTS2019Train/ all(all is for fold 0~4;if you want to train fold 0,you can set 0)
There is a Temporary Workaround in current version of NPU,The path is depand on the installation path of python.
- sed -i 's#'sync_mode=2'#'sync_mode=0'#g' /usr/local/lib/python3.7/site-packages/te/platform/cce_build.py
	
Examples:
- Case for training performance on a single NPU
    - In *train_1p_performance.sh*, python scripts part should look like as follows. For more detailed command lines arguments, please refer to [Command-line options](#command-line-options)
        ```shell
        python3.7 ${currentDir}/main_npu.py \
          --data_dir=/data/BraTS2019Train \
          --model_dir=result \
          --exec_mode=train \
          --max_steps=80 \
          --benchmark \
          --fold=0 \
          --batch_size=2 \
          --augment 
        ```
    - Run the program  
        ```
        bash run_1p_performance.sh
        ```
- Case for training performance on 8 NPUs
    - In *train_8p_performance.sh*, python scripts part should look like as follows.
        ```shell 
        python3 ${currentDir}/main_npu.py \
          --data_dir=/data/BraTS2019Train \
          --model_dir=result \
          --exec_mode=train \
          --max_steps=80 \
          --benchmark \
          --fold=0 \
          --batch_size=2 \
          --augment 
        ```
    - Run the program  
        ```
        bash run_8p_performance.sh
        ```
- Case for training accuracy on 8 NPUs
    - In *train_8p_accuracy.sh*, python scripts part should look like as follows.
        ```shell 
        python3 ${currentDir}/main_npu.py \
          --data_dir=/data/BraTS2019Train \
          --model_dir=result \
          --exec_mode=train_and_evaluate \
          --max_steps=16000 \
		  --npu_loss_scale=1 \
          --augment \
          --batch_size=2 \
          --fold=${fold} 
        ```
    - Run the program  
        ```
        bash run_8p_accuracy.sh /npu/z00438116/BraTS2019Train/ all
        ```

### 5. Test
- In *run_eval.sh*, python scripts part should look like as follows:
     ```shell 
    python3.7 ${currentDir}/main_npu.py \
      --data_dir=/data/BraTS2019Train \
      --model_dir=result/1p/0/result \
      --exec_mode=evaluate \
      --fold=0 \
      --batch_size=1
    ```
    Remember to modify the `data_dir` and `model_dir`, then run the program  
    ```
    bash run_eval.sh
    ```


## Advanced
### Command-line options

We list those important parameters to train this network here. For more details of all the parameters, please read *main_npu.py* and other related files.

```
  --data_dir                        directory of dataset (required)
  --model_dir                       directory where the model stored (required)
  --exec_mode                       mode to run the code (default: None)
  --max_steps                       maximum number of steps for training (default: 16000)
  --benchmark                       enable performance benchmarking (default: False)
  --fold                            select fold for cross-validation (default: 0)
  --batch_size                      size of each minibatch (default: 1)
  --augment                         enable data augmentation (default: False)
```

### Training process

After training, all the results of the training will be stored in the directory `result`.
 
## Performance

### Result

Our results were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **steps** |    TumorCore   |   PeritumoralEdema   |   EnhancingTumor   |   MeanDice   |   WholeTumor   |
| :-------: | :------------: | :------------------: | :----------------: | :----------: | :------------: |
|   16000   |     0.6532     |        0.7555        |      0.7085        |    0.7057    |     0.8825     |

#### Training performance results
| **NPUs** |  throughput_train  |
| :------: | :----------------: |
|    1     |    2.28 images/s   |

| **NPUs** |  throughput_train |
| :------: | :---------------: |
|    8     |   18.27 images/s  |











