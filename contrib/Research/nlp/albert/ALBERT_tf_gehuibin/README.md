迁移[Albert](https://github.com/google-research/albert)到ascend910平台
使用的是albert_v2版本的预训练模型
|  | F1| EM |
| :-----| ----: | :----: |
| albert_base(Ascend) | 82.4| 79.4|
| albert_base(论文) | 82.1 | 79.3 |
| albert_large(Ascend) | 84.2 | 81.3 |
| albert_large(论文) | 84.9 | 81.8 |


训练和预测脚本:

albert_base
```
./squad2_base.sh
```
albert_large
```
./squad2_large.sh
```
如果只训练则注释掉--do_train

只预测则注释掉--do_predict

输入文件需要建立squad_v2文件夹

对于albert_base需要建立albert_base_v2, output_base_v2文件夹

对于albert_large需要建立albert_base_v2, output_base_v2文件夹

上述文件夹均可从

[百度网盘](https://pan.baidu.com/s/1F_8A398wefDj9woOJ71MwQ)提取码: 7taq 下载


# ALBERT
## 概述
This example implements training and evaluation of Transformer Model, which is introduced in the following paper:
- Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, Ł ukaszKaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS 2017, pages 5998–6008.

## Requirements
- Tensorflow 1.15.0.
- Download and preprocess the WMT English-German dataset for training and evaluation.

>  Notes:If you are running an evaluation task, prepare the corresponding checkpoint file.

## 代码路径解释

```shell
.
└─ 
  ├─README.md
  ├─output_base_v2 基于squadv2微调过的albert base模型路径
  	├─checkpoint
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  	└─...
  ├─output_large_v2 基于squadv2微调过的albert base模型路径
  	├─checkpoint
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta
  	└─...
  ├─albert_base_v2 albert base的预训练模型
  	├─30k-clean.model
  	├─30k-clean.vocab
  	├─albert_config.json
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta

  ├─albert_large_v2 albert large的预训练模型
  	├─30k-clean.model
  	├─30k-clean.vocab
  	├─albert_config.json
  	├─model.ckpt-best.data-00000-of-00001
  	├─model.ckpt-best.index
  	├─model.ckpt-best.meta

  ├─squad2_base.sh albert base的启动脚本
  ├─squad2_large.sh albert large的启动脚本
  └─train-ende.sh
```

---

## Prepare the dataset
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

## Running the example
modify the permission of the script to 	be run in the file
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

---

## Usage

## Parameters
It contains of parameters of Transformer model and options for training and evaluation, which is set in file `train-ende.sh`.
### Parameters:
```
Parameters for training (Training/Evaluation):
    batch_size                      batch size of input dataset: N, default is 40
    max_length                      length of input sequence: N, default is 128
    train_steps						totally training steps for training: N, default is 300000
    dropout_rate             		dropout probability for TransformerOutput: Q, default is 0.2
    optimizer.name					optimizer used in the network: Adam, default is "Adam"
    learning_rate.warmup_steps		value of learning rate: N
    init_loss_scale                 initial value of loss scale: N, default is 2^10
```