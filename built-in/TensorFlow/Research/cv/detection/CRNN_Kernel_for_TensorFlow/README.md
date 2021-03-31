# CRNN for Tensorflow 

This repository provides a script and recipe to train the CRNN  model. The code is based on https://github.com/MaybeShewill-CV/CRNN_Tensorflow,
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

CRNN model from
`Shibao Guang  et al. "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition". <https://arxiv.org/abs/1507.05717>.`
reference implementation:  <https://github.com/MaybeShewill-CV/CRNN_Tensorflow>
### Model architecture



### Default configuration

The following sections introduce the default configurations and hyperparameters for CRNN  model. We reproduce training setups 
on synth90k datasets, evaluate on three datasets: IIIT5K, ICDAR2003 and SVT. See [Results](#results) for details.

For detailed hpyerparameters, please refer to corresponding scripts under directory `scripts/` and 
#### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Momentum : 0.9
- LR schedule: cosine_annealing
- Batch size : 64 * 8   

#### Data augmentation

This model uses the following data augmentation:

- For training:
  - Normalize=(value/127.5-1.0)
- For inference:
  - Normalize=(value/127.5-1.0)



## Setup
The following section lists the requirements to start training the CRNN model.
### Requirements

see `requirements.txt`

## Quick Start Guide

### 1. Clone the respository

```shell
git clone xxx
cd  ModelZoo_CRNN_TF/
```

### 2. Download and preprocess the dataset

You can use any datasets as you wish. Here, we only  synth90k dataset as an example to illustrate the data generation. 

1. Download the synth90k, IIIT5K, ICDAR2003 and SVT datasets and put them under `./data`. 
2. go to  `/data` directory and unzip the datasets


```
bash prepare_ds.sh
``` 
After data preparation, the directory of  `data/` looks like following structure:

|-- data/
|     |-- char_dict/
|     |-- mnt/
|     |-- images/
|     |-- test/
|     |-- tfrecords/


### 3. Train

All the scripts to tick off the training are located under `scripts/`. Make sure that all data are ready before you start training. Training on single NPU or multiple NPU devices are supported. Scripts that contain `1p` indicate single NPU training scripts or configuration. Scripts that contain `8p` indicate training on eight NPU devices.
   
- For training on single NPU device, execute the shell ModelZoo_CRNN_TF `run_1p.sh`, e.g.
  ```
   bash run_1p.sh
  ```
   By default, the checkpoints and training log are located in `./`.

- For training on eight NPU device, execute the shell script `run_8p.sh`, e.g.
  ```
   bash /run_8p.sh 2000 0.002 20
  ```
  By default, the checkpoints and training log are located in `results/8p/`. 


***Note***: As the time consumption of the training for single NPU is much higher than that of 8 NPUs, it is recommended to train on eight NPUs.


### 4. Test
Three datases are used to evaluate the trained model. To test, just run test script 'scripts/test.sh ${DIR_TO_CHECKPOINTS}' (replace the ${DIR_TO_CHECKPOINTS}  with real path to checkpoint file). When finished, test results will be saved as text file under project directory with name `test_result.txt` by default.
  ```
   bash test_ckpt.sh ${DIR_TO_CHECKPOINTS}
  ```


## Advanced
### Commmand-line options


```
  --root_dir                        Root directory of the project, default ./
  --dataset_dir                     path to tfrecords file, default data/
  --weights_path	            pretrained checkpoint when continuing training, default None
  --momentum                        momentum factor, default: 0.9
  --num_iters                       the number of training steps , default 240000
  --lr_sched                        the lr scheduling policy, default cosine
  --use_nesterov                    whether to use nesterov in the sgd optimizer, default ,False
  --warmup_step                     number of warmup step used in lr schedular                    
```
for a complete list of options, please refer to `tools/train_npu.py` and `config/global_config.py`

### Training process

All the results of the training will be stored in the directory `results`.
Script will store:
 - checkpoints
 - log
 
## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.


#### Evaluation results 
The accuracy is measured in term of full sequence in a lexicon-free decoding mode 

| **training  steps**| SVT         | ICDAR2003   | IIIT5k     |
| :----------------: | :----------:| :------:    |:------:    |
|    8               |  80.8% +    |  89.4%+     | 78.2% +    |


#### Training performance 

| **NPUs** | batch size        | train performance |
| :------: | :---------------: |:---------------:  |
|    8     | 64*8              |  ~ 168ms/step     |











