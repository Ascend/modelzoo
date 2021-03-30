recommendation Model
## Overview
This is an implementation of WideDeep as described in the [Wide & Deep Learning for Recommender System](https://arxiv.org/pdf/1606.07792.pdf) paper. 

WideDeep model jointly trained wide linear models and deep neural network, which combined the benefits of memorization and generalization for recommender systems. 

## Dataset
The [Criteo datasets](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/) are used for model training and evaluation.

## Running Code

### Download and preprocess dataset
To download the dataset, please install Pandas package first. Then issue the following command:
```
python scripts/process_data.py
```

### Code Structure
The entire code structure is as following:
```
    1, src/  
        callback.py            "Callback class: EvaluateCallback, LossCallback;"
        config.py              "config args for data, model, train"
        datasets.py            "dataset loader class"
        WideDeep.py            "Model structure code, include: DenseLayer, widedeepModel, NetWithLossClass, TrainStepWrap, PredictWithSigmoid, ModelBuilder"

    2, ./
        train.py               "The main script for train and eval, init by the config.py in deepfm_model_zoo/src/config.py "
        test.py                "The main script for predict, load checkpoint file."
        rank_table_2p.json     "The config file for two npus"
        rank_table_4p.json     "The config file for four npus"
        rank_table_8p.json     "The config file for eight npus"

    3, scripts/
        process_data.py        "The file for raw data download and process"
        run_train.sh           "The shell script for training"
        run_eval.sh            "The shell script for evaluation"
    4, scripts_16p/
        start.sh               "the run file of sixteen npus"     
    5, tools/
        cluster_16p.json       "the info of npu"      
        common.sh              "the ssh key config of npu"
        hccl.json              "the config file of npu"
```

### Train and evaluate model
To train and evaluate the model, issue the following command:
```
python tools/train_and_test.py
```
Arguments:
  * `--data_path`: This should be set to the same directory given to the data_download's data_dir argument.
  * `--epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--field_size`: The number of features.
  * `--vocab_size`： The total features of dataset.
  * `--emb_dim`： The dense embedding dimension of sparse feature.
  * `--deep_layers_dim`： The dimension of all deep layers.
  * `--deep_layers_act`： The activation of all deep layers.
  * `--keep_prob`： The rate to keep in dropout layer.
  * `--ckpt_path`：The location of the checkpoint file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.

To train the model, issue the following command:
```
python tools/train.py
```
Arguments:
  * `--data_path`: This should be set to the same directory given to the data_download's data_dir argument.
  * `--epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--field_size`: The number of features.
  * `--vocab_size`： The total features of dataset.
  * `--emb_dim`： The dense embedding dimension of sparse feature.
  * `--deep_layers_dim`： The dimension of all deep layers.
  * `--deep_layers_act`： The activation of all deep layers.
  * `--keep_prob`： The rate to keep in dropout layer.
  * `--ckpt_path`：The location of the checkpoint file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.

To evaluate the model, issue the following command:
```
python tools/test.py
```
Arguments:
  * `--data_path`: This should be set to the same directory given to the data_download's data_dir argument.
  * `--epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--field_size`: The number of features.
  * `--vocab_size`： The total features of dataset.
  * `--emb_dim`： The dense embedding dimension of sparse feature.
  * `--deep_layers_dim`： The dimension of all deep layers.
  * `--deep_layers_act`： The activation of all deep layers.
  * `--keep_prob`： The rate to keep in dropout layer.
  * `--ckpt_path`：The location of the checkpoint file.
  * `--eval_file_name` : Eval output file.
  * `--loss_file_name` :  Loss output file.

There are other arguments about models and training process. Use the `--help` or `-h` flag to get a full list of possible arguments with detailed descriptions.

