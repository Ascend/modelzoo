# Transformer for Tensorflow

## Table Of Contents

* [Description](#Description)
* [Requirements](#Requirements)
* [Default configuration](#Default-configuration)
  * [Optimizer](#Optimizer)
* [Quick start guide](#quick-start-guide)
  * [Prepare the dataset](#Prepare-the-dataset)
  * [Docker container scene](#Docker-container-scene)
  * [Key configuration changes](#Key-configuration-changes)
  * [Running the example](#Running-the-example)
    * [Training](#Training)
    * [Training process](#training-process)
    * [Evaluation](#Evaluation)
* [Advanced](#advanced)
  * [Commmand-line options](#Commmand-line-options)

## Description

This example implements training and evaluation of Transformer Model, which is introduced in the following paper:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## Requirements

- Install Tensorflow 1.15.0.
- Download and preprocess the WMT English-German dataset for training and evaluation.

>  Notes:If you are running an evaluation task, prepare the corresponding checkpoint file.

## Default configuration

The following sections introduce the default configurations and hyperparameters for Transformer model.

### Optimizer

- Learning rate(LR): 2.0
- Batch size: 40
- Label smoothing: 0.1
- num_units: 1024
- num_layers: 6
- attention.num_heads: 16

## Quick start guide

### Prepare the dataset

- You can download and preprocess WMT English-German dataset by yourself. Assuming you get the following files:
  - train.tok.clean.bpe.32000.en
  - train.tok.clean.bpe.32000.de
  - vocab.share
  - newstest2014.tok.bpe.32000.en
  - newstest2014.tok.bpe.32000.de
  - newstest2014.tok.de
- Convert the original data to mindrecord for training:

    ``` bash
    paste train.tok.clean.bpe.32000.en train.tok.clean.bpe.32000.de > train.all
    python create_training_data_concat.py --input_file train.all --vocab_file vocab.bpe.32000 --output_file /path/ende-l128-mindrecord --max_seq_length 128
    ```
- Convert the original data to mindrecord for evaluation:

    ``` bash
    paste newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de > test.all
    python create_training_data_concat.py --input_file test.all --vocab_file vocab.bpe.32000 --output_file /path/newstest2014-l128-mindrecord --num_splits 1 --max_seq_length 128 --clip_to_max_len True
    ```

### Docker container scene

- Compile image
```bash
docker build -t ascend-transformer .
```

- Start the container instance
```bash
bash docker_start.sh
```

Parameter Description:

```bash
#!/usr/bin/env bash
docker_image=$1 \   #Accept the first parameter as docker_image
data_dir=$2 \       #Accept the second parameter as the training data set path
model_dir=$3 \      #Accept the third parameter as the model execution path
docker run -it --ipc=host \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \  #The number of cards used by docker, currently using 0~7 cards
        --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v ${data_dir}:${data_dir} \    #Training data set path
        -v ${model_dir}:${model_dir} \  #Model execution path
        -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
        -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \     #docker_image is the image name
        /bin/bash
```

After executing docker_start.sh with three parameters:
  - The generated docker_image
  - Dataset path
  - Model execution path
```bash
./docker_start.sh ${docker_image} ${data_dir} ${model_dir}
```


### Key configuration changes

Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


### Running the example

#### Training

- Set basic configs in `configs/transformer_big.yml`, including model_params, learning rate and network hyperparameters. 

- Set options in `train-ende.sh`, including batch_size, data path and training hyperparameters. 

- Run `transformer_1p/transformer_main_1p.sh` for non-distributed training of Transformer model.

    ``` bash
    sh transformer_main_1p.sh
    ```

- Run `transformer_8p/transformer_8p.sh` for distributed training of Transformer model.

    ``` bash
    sh transformer_8p.sh
    ```

#### Training process

All the results of the training will be stored in the directory `model_dir`.
Script will store:
 - checkpoints
 - log

#### Evaluation

- Set options in `inference.sh`. Make sure the 'DATA_PATH', 'TEST_SOURCES', 'MODEL_DIR' and "output" are set to your own path.

- Run `inference.sh` for evaluation of Transformer model.

    ```bash
    ./inference.sh
    ```

- Run `process_output.sh` to process the output token ids to get the real translation results.

- REF_DATA is the target text, EVAL_OUTPUT is the output of inference.sh, VOCAB_FILE is the vocab file in the dataset. (In our example, REF_DATA is  newstest2014.tok.bpe.32000.de, VOCAB_FILE  is vocab.share)

    ```bash
    sh scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE
    ```
    You will get two files, REF_DATA.forbleu and EVAL_OUTPUT.forbleu, for BLEU score calculation.

- Calculate BLEU score, go to directory `ModelZoo_Transformer_TF/` and run following command to get the BLEU score.

    ```bash
    cd ModelZoo_Transformer_TF
    perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
    ```

#### ckpt to pb

- run frozen_graph.sh to generate transformer pb model.

```
./frozen_graph.sh

these parameters needs to be modified:
MODEL_DIR="./model_dir_base"    ---- the path of ckpt
vocab_source:/home/wmt-ende/vocab.share  --- the path of vocab dic
vocab_target:/home/wmt-ende/vocab.share  --- the path of vocab dic
output_filename:"infer-b${beam}.pb"   --- the pb model name
 remark:the time for ckpt to pb is about 1h.
```



## Advanced

It contains of parameters of Transformer model and options for training and evaluation, which is set in file `train-ende.sh`.

### Commmand-line options

```
Parameters for training (Training/Evaluation):
    --config_paths                 Path for training config file.
    --model_params                 Model parameters for training the model, like learning_rate,dropout_rate.
    --metrics                      Metrics for model eval.
    --input_pipeline_train         Dataset input pipeline for training.
    --input_pipeline_dev           Dataset input pipeline for dev.
    --train_steps                  Training steps, default is 300000.
    --keep_checkpoint_max          Number of the checkpoint kept in the checkpoint dir.
    --batch_size                   Batch size for training, default is 40.
    --model_dir                    Model dir for saving checkpoint
    --use_fp16                     Whether to use fp16, default is True.
```