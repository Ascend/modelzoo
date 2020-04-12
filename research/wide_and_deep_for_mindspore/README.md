# Recommendation Model
## Overview

This is an implementation of WideDeep as described in the [Wide & Deep Learning for Recommender System](https://arxiv.org/pdf/1606.07792.pdf) paper.

WideDeep model jointly trained wide linear models and deep neural network, which combined the benefits of memorization and generalization for recommender systems.

## Dataset
The [Criteo datasets](http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/) are used for model training and evaluation.

## Running Code

### Download and preprocess dataset
To download the dataset, please install Pandas package first. Then issue the following command:
```
bash download.sh
```

### Code Structure
The entire code structure is as following:
```
    |--- scripts/                "demo bash of training and evaluation"
        run_train.sh             "Wide&Deep model training bash"
        run_eval.sh              "Wide&Deep model evaluation bash"
        run_train_and_test.sh    "Wide&Deep model training and evaluation bash"
        download.sh              "dataset download bash"
    |--- tools/                  "entrance of training and evaluation"
        config.py                "parameters configuration"
        test.py                  "Entrance of Wide&Deep model evaluation"
        train.py                 "Entrance of Wide&Deep model training"
        train_and_test.py        "Entrance of Wide&Deep model training and evaluation"
    |--- wide_deep/              "Core implementation of Wide&Deep model"
        |--- data/
            dataset.py           "Dataset loader class"
        |--- models/
            WideDeep.py          "Model structure"
        |--- utils/
            callbacks.py         "Callback class for training and evaluation"
            metrics.py           "Metric class"
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
