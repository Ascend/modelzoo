# Music Auto Tagging for MindSpore

This repository provides a script and recipe to train the Music Auto Tagging model to achieve state-of-the-art accuracy.

## Table Of Contents

- Model overview

  - Default configuration

    - Optimizer
  
- Setup

  - Requirements

- Quick Start Guide

- Performance

  - Results

## Model overview

Music Auto Tagging is a convolutional neural network architecture, its name Music Auto Tagging comes from the fact that it has 4 layers. Its layers consists of Convolutional layers, Max Pooling layers, Activation layers, Fully connected layers.

VGG16 model from
    `"Keunwoo Choi, George Fazekas, and Mark Sandler, “Automatic tagging using deep convolutional neural networks,” in International Society of Music Information Retrieval Conference. ISMIR, 2016." <https://arxiv.org/abs/1606.00298>`
### Default configuration

The following sections introduce the default configurations and hyperparameters for Music Auto Tagging model.

#### Optimizer

This model uses Adam optimizer from mindspore with the following hyperparameters:

- Learning rate (LR) : 0.0005
- Loss scale: 1024.0
- Batch size : 32
- We train for:
  - 50 epochs for a standard training process

## Setup

### Requirements

Ensure you have the following components:
  - [MindSpore](https://www.mindspore.cn/)
  - Hardware environment with the Ascend AI processor


  For more information about how to get started with MindSpore, see the
  following sections:
  - [MindSpore's Tutorial](https://www.mindspore.cn/tutorial/zh-CN/master/index.html)
  - [MindSpore's Api](https://www.mindspore.cn/api/zh-CN/master/index.html)


## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd modelzoo_fcn4
```

### 2. Download and preprocess the dataset

1. down load the classification dataset
2. Extract the training data
3. The train and val images are under the train/ and val/ directories, respectively. Labels of each clip contain in information file.

### 3. Train

after having your dataset, first convert the audio clip into mindrecord dataset by using the following codes
```shell
# npu: device id
# get_npy: mode for converting to npy, default 1 in this case
# get_mindrecord: mode for converting npy file into mindrecord file，default 1 in this case
# info_path: directory of label.csv, which provide the label of each audio clips
# info_name: name of label data
# file_path: directory of audio clips
# npy_path: directory of saved numpy
# mr_info: tmp file for generating train and val file
# mr_path: directory for saving mindrecord file
# mr_name: name of mindrecord, generate as {mr_name}.mindrecord0
# num_classes: number of tagging classes
python data_conversion.py --npu 0 \                          
                          --get_npy 1 \                          
                          --get_mindrecord 1 \                          
                          --info_path ../data/ \                          
                          --info_name label.csv \                          
                          --file_path ../data/audio/ \                          
                          --npy_path ../data/npy/                          
                          --mr_info ../data/train.csv \                          
                          --mr_path ../data/fea/ \                          
                          --mr_name train \                          
                          --num_classes 50
```

Then, you can start training the model by using the following codes
```shell
# npu: device id
# epoch: training epoch
# batch: batch size
# lr: learning rate
# ls: loss scale
# mixed_precision: if use mixed mode or not
# data_dir: directory of mindrecord data
# filename: file name of the mindrecord data
# num_consumer: file number for mindrecord 4
# prefix: prefix of checkpoint
# save_step: steps for saving checkpoint
# model_dir: directory of model
# model_name: load model name, if it is set, srcipt will keep training based on this model
SLOG_PRINT_TO_STDOUT=0 python train.py --npu 0 \                                       
                                       --epoch 50 \                                       
                                       --batch 32 \                                       
                                       --lr 0.0005 \                                       
                                       --ls 1024.0 \                                       
                                       --data_dir ../data/fea/ \                                       
                                       --filename train.mindrecord0 \                                      
                                       --num_consumer 4 \                                       
                                       --prefix MusicTagger \                                       
                                       --save_step 2000 \                                       
                                       --model_dir ../data/fea/model/ \                                       
                                       --mixed_precision True
```

### 4. Test
Again, you need to convert the test data into mindrecord dataset

```shell
python data_conversion.py --npu 0 \                          
                          --get_npy 0 \                          
                          --get_mindrecord 1 \                          
                          --npy_path ../data/npy/ \                          
                          --mr_info ../data/val.csv \                           
                          --mr_path ../data/fea/ \                           
                          --mr_name val
```

Then you can test your model
```shell
SLOG_PRINT_TO_STDOUT=0 python test.py --npu 0 \                                       
                                      --model_dir ../data/fea/model/ \                                       
                                      --model_name MusicTagger-48_479.ckpt \                                       
                                      --batch 32 \                                       
                                      --data_dir ../data/fea/ \                                       
                                      --filename val.mindrecord0 \                                       
                                      --num_consumer 4
```

### 5. Tagging

Finally, you can use the moedl to tag your own music
```shell
SLOG_PRINT_TO_STDOUT=0 python demo.py --npu 0 \                                      
                                      --model_dir ../data/fea/model/ \                                      
                                      --model_name MusicTagger-48_479.ckpt \                                      
                                      --audio_file "./clips/4-01.Beethoven-Symphony No. 3 in E-flat - Allegro con brio.wav"
```

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Training accuracy results

| **epochs** |   Loss   |
| :--------: | :-----------: |
|    50     | 0.11 |

#### Training performance results

| **GPUs** | train performance |
| :------: | :---------------: |
|    1     |   160  samples/s   |

#### Testing accuracy results

| **epochs** |   ROC   |
| :--------: | :-----------: |
|    -     | 0.9099 |

#### Testing performance results

| **GPUs** | train performance |
| :------: | :---------------: |
|    1     |   225.0  samples/s   |