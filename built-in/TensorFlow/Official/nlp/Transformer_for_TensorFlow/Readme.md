# Transformer for Tensorflow
## Description
This example implements training and evaluation of Transformer Model, which is introduced in the following paper:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## Requirements
- Install Tensorflow 1.15.0.
- Download and preprocess the WMT English-German dataset for training and evaluation.

>  Notes:If you are running an evaluation task, prepare the corresponding checkpoint file.

## Quick Start Guide
### Prepare the dataset
- You may use this [shell script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh) to download and preprocess WMT English-German dataset. Assuming you get the following files:
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

### Key configuration changes
Before starting the training, first configure the environment variables related to the program running. For environment variable configuration information, see:
- [Ascend 910 environment variable settings](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


## Running the example
### Training
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

### Evaluation
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

- Calculate BLEU score, you may use this [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) and run following command to get the BLEU score.

    ```bash
    perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
    ```

## Parameters
It contains of parameters of Transformer model and options for training and evaluation, which is set in file `train-ende.sh`.
### Parameters:
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