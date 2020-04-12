## Overview
### Prepare the dataset
===============

1. Download and preprocess the WMT English-German dataset:

    https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh

    Assuming we get the following files:
    * data/train.tok.clean.bpe.32000.en
    * data/train.tok.clean.bpe.32000.de
    * data/vocab.bpe.32000

2. Convert the original data to tf record data:

```
paste data/train.tok.clean.bpe.32000.en data/train.tok.clean.bpe.32000.de > data/train.all

python transformer/create_training_data.py --input_file data/train.all --vocab_file data/vocab.bpe.32000 --output_file data/tfrecord/ende-l128-tfrecord
```

Now you can follow the steps in ``Running the code'' section to run our experiments.

---

## Running the code
### Training

The code sample in this directory uses MindSpore API.

You can run the code as follows:

* single card:

    ```
    sh -x scripts/run_1p.sh
    ```

    The model is saved to `run_single/model_dir/` by default, which can be changed using the `--checkpoint_path` flag in `scripts/run_1p.sh`.

* or 8 cards (recommended):

    ```
    sh -x scripts/run_8p.sh
    ```

    The model is saved to `helperx/model_dir/` by default, which can be changed using the `--checkpoint_path` flag in `scripts/parallel_train.sh`.

    The file "scripts/hccl8.json" is an example configuration for using 8 cards. You may need to modify it according to your devices.

### Parameters

You can set the the following running parameters in the script``:

* --data_path: The location of the input tfrecord data.
* --train_epochs: The number of epochs used to train.
* --batch_size: The batch size used to train.
* --checkpoint_path: The location of the checkpoint file.

Model configurations can be changed in the `get_config` function of the `train_main.py`. Some of the key parameters are as follows:

* seq_length: sentence length of the dataset,
* vocab_size: the size of vocab file, multiple of 16 is recommended,
* hidden_size: size of the layers,
* num_hidden_layers: the number of encoder and decoder layers,
* num_attention_heads: the number of heads in Multihead Attetnion,
* intermediate_size: the hidden size of the FeedForward layer,
* hidden_act="relu": activation function of the FeedForward layer,
* hidden_dropout_prob: dropout rate of layer outputs,
* attention_probs_dropout_prob: dropout rate of the attention probs in Multhead Attention,
* max_position_embeddings: the maximum positions of position encoding,
* compute_type: the precision for calculation, mstype.float16 is recommended for faster training.
