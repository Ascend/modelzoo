# Contents

- [Contents](#contents)
- [TinyBERT Description](#tinybert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)
    - [General Distill](#general-distill)
    - [Task Distill](#task-distill)
  - [Options and Parameters](#options-and-parameters)
    - [Options:](#options)
    - [Parameters:](#parameters)
  - [Training Process](#training-process)
    - [Notice](#notice)
      - [Weight transforming of teacher](#weight-transforming-of-teacher)
      - [The difference of training of task distill](#the-difference-of-training-of-task-distill)
      - [Places to be modified when finetuning bert model](#places-to-be-modified-when-finetuning-bert-model)
    - [Training](#training)
      - [running on Ascend](#running-on-ascend)
      - [running on GPU](#running-on-gpu)
    - [Distributed Training](#distributed-training)
      - [running on Ascend](#running-on-ascend-1)
      - [running on GPU](#running-on-gpu-1)
  - [Evaluation Process](#evaluation-process)
    - [Evaluation](#evaluation)
      - [evaluation on SST-2 dataset](#evaluation-on-sst-2-dataset)
      - [evaluation on MNLI dataset](#evaluation-on-mnli-dataset)
      - [evaluation on QNLI dataset](#evaluation-on-qnli-dataset)
  - [Model Description](#model-description)
  - [Performance](#performance)
    - [training Performance](#training-performance)
      - [Inference Performance](#inference-performance)
    - [TinyBert Eval in device](#tinybert-eval-in-device)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [TinyBERT Description](#contents)

[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) is 7.5x smalller and 9.4x faster on inference than [BERT-base](https://github.com/google-research/bert) (the base version of BERT model) and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages.

[Paper](https://arxiv.org/abs/1909.10351): Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). arXiv preprint arXiv:1909.10351.

# [Model Architecture](#contents)

The backbone structure of TinyBERT is transformer, the transformer contains four encoder modules, one encoder contains one selfattention module and one selfattention module contains one attention module.

# [Dataset](#contents)

- Download the zhwiki or enwiki dataset for general distillation. Extract and clean text in the dataset with [WikiExtractor](https://github.com/attardi/wikiextractor). Convert the dataset to TFRecord format, please refer to create_pretraining_data.py which in [BERT](https://github.com/google-research/bert) repository.

  - Notice(processing of enwiki): when extract enwiki, try to slice the dataset to a smaller size, to avoid memory overflowing. It's better to use scripts to handle slices accordingly to accelerate.

    ```
    git clone https://github.com/attardi/wikiextractor
    python WikiExtractor.py -b 128M -o ../../extracted ../../enwiki-latest-pages-articles.xml.bz2
    ```

    To find vocab_file in google's bertrepo, find coresponding checkpoints in bert-base.

    ```
    sudo pip install bert-tensorflow
    python create_pretraining_data.py --input_file=./enwiki/extracted/AA/wiki_00 --output_file=./enwiki/tfrecord/enwiki_00.tfrecord --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5
    ```

    Due to the many slices of file and long processing time, cache problems may occur during merging. It's better to handle data one by one, you can refer to the transfroming scripts bellow.

    ```
    for i in {17..24}
    do
        python create_pretraining_data.py --input_file=./enwiki/extracted/AA/wiki_${i} --output_file=./enwiki/tfrecord/enwiki_${i}.tfrecord --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5  &
    done
    ```

    Since the transform scripts is single-threaded, if you are on server and have a memory large enough, you can choose to run several scripts at the same time, decompose and transform the dataset, the main place you need to change is in the for loop {17..24}

    Note that if bert-base is trained, more that 100G of data is needed. If tiny-bert is trained, only around 10G of data is needed.

- Download glue dataset for task distillation. Convert dataset files from json format to tfrecord format, please refer to run_classifier.py which in [BERT](https://github.com/google-research/bert) repository.

  - Notice（processing of SST-2）: This is the standard dataset of glue benchmark. Add the following code in run_classifier.py. **The script is in tools/gluebenchmark_convert_record.**

    ```
    class Sst2Processor(DataProcessor):
      """Processor for the SST-2 data set (GLUE version)."""

      def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

      def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

      def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

      def get_labels(self):
        """See base class."""
        return ["0", "1"]

      def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          if i == 0:
            continue
          guid = "%s-%s" % (set_type, i)
          text_a = tokenization.convert_to_unicode(line[0])
          if set_type == "test":
            label = "0"
          else:
            label = tokenization.convert_to_unicode(line[1])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    processors = {
        "cola": ColaProcessor,
        "sst2": Sst2Processor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
    }
    ```

    And then use the following command.

    ```
    # your dataset path
    export BERT_BASE_DIR=/home/admin/dataset/cased_L-12_H-768_A-12
    export GLUE_DIR=/home/admin/dataset/SST-2

    python run_classifier.py \
      --task_name=SST2 \
      --do_eval=true \
      --data_dir=$GLUE_DIR/SST-2 \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --do_lower_case=False \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=./SST-2/tfrecord/
    ```

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU）
  - Prepare hardware environment with Ascend or GPU processor. If you want to try Ascend, please send the [application form](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx) to ascend@huawei.com. Once approved, you can get the resources.
- Framework
  - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start general distill, task distill and evaluation as follows:

```bash
# run standalone general distill example
bash scripts/run_standalone_gd.sh

Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_standalone_gd.sh file first. If running on GPU, please set the `device_target=GPU`.

# For Ascend device, run distributed general distill example
bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json

Before running the shell script, please set the `load_teacher_ckpt_path`, `data_dir`, `schema_dir` and `dataset_type` in the run_distributed_gd_ascend.sh file first.

# For GPU device, run distributed general distill example
bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt

# run task distill and evaluation example
bash scripts/run_standalone_td.sh

Before running the shell script, please set the `task_name`, `load_teacher_ckpt_path`, `load_gd_ckpt_path`, `train_data_dir`, `eval_data_dir`, `schema_dir` and `dataset_type` in the run_standalone_td.sh file first.
If running on GPU, please set the `device_target=GPU`.
```

For distributed training on Ascend, a hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

for example,the hccl json format(8P,modified the "device_num/instance_count/instance_list" when your device number different) :
```
{
    "board_id": "0x0020",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0",
        "eth1",
        "eth2",
        "eth3",
        "eth4",
        "eth5",
        "eth6",
        "eth7"
    ],
    "para_plane_nic_num": "8",
    "status": "completed",
    "group_list": [
        {
            "device_num": "8",
            "server_num": "1",
            "group_name": "test",
            "instance_count": "8",
            "instance_list": [
                {
                    "devices": [
                        {
                            "device_id": "0",
                            "device_ip": "192.168.10.11"
                        }
                    ],
                    "rank_id": "0",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "1",
                            "device_ip": "192.168.20.11"
                        }
                    ],
                    "rank_id": "1",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "2",
                            "device_ip": "192.168.30.11"
                        }
                    ],
                    "rank_id": "2",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "3",
                            "device_ip": "192.168.40.11"
                        }
                    ],
                    "rank_id": "3",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "4",
                            "device_ip": "192.168.10.12"
                        }
                    ],
                    "rank_id": "4",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "5",
                            "device_ip": "192.168.20.12"
                        }
                    ],
                    "rank_id": "5",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "6",
                            "device_ip": "192.168.30.12"
                        }
                    ],
                    "rank_id": "6",
                    "server_id": "10.34.0.4"
                },
                {
                    "devices": [
                        {
                            "device_id": "7",
                            "device_ip": "192.168.40.12"
                        }
                    ],
                    "rank_id": "7",
                    "server_id": "10.34.0.4"
                }
            ]
        }
    ]
}

```

For dataset, if you want to set the format and parameters, a schema configuration file with JSON format needs to be created, please refer to [tfrecord](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/dataset_loading.html#tfrecord) format.

```
For general task, schema file contains ["input_ids", "input_mask", "segment_ids"].

For task distill and eval phase, schema file contains ["input_ids", "input_mask", "segment_ids", "label_ids"].

`numRows` is the only option which could be set by user, the others value must be set according to the dataset.

For example, the dataset is cn-wiki-128, the schema file for general distill phase as following:
{
	"datasetType": "TF",
	"numRows": 7680,
	"columns": {
		"input_ids": {
			"type": "int64",
			"rank": 1,
			"shape": [256]
		},
		"input_mask": {
			"type": "int64",
			"rank": 1,
			"shape": [256]
		},
		"segment_ids": {
			"type": "int64",
			"rank": 1,
			"shape": [256]
		}
	}
}
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─bert
  ├─README.md
  ├─scripts
    ├─run_distributed_gd_ascend.sh       # shell script for distributed general distill phase on Ascend
    ├─run_distributed_gd_gpu.sh          # shell script for distributed general distill phase on GPU
    ├─run_standalone_gd.sh               # shell script for standalone general distill phase
    ├─run_standalone_td.sh               # shell script for standalone task distill phase
  ├─src
    ├─__init__.py
    ├─assessment_method.py               # assessment method for evaluation
    ├─dataset.py                         # data processing
    ├─gd_config.py                       # parameter configuration for general distill phase
    ├─td_config.py                       # parameter configuration for task distill phase
    ├─tinybert_for_gd_td.py              # backbone code of network
    ├─tinybert_model.py                  # backbone code of network
    ├─utils.py                           # util function
  ├─__init__.py
  ├─run_general_distill.py               # train net for general distillation
  ├─run_task_distill.py                  # train and eval net for task distillation
```

## [Script Parameters](#contents)

### General Distill

```
usage: run_general_distill.py   [--distribute DISTRIBUTE] [--epoch_size N] [----device_num N] [--device_id N]
                                [--device_target DEVICE_TARGET] [--do_shuffle DO_SHUFFLE]
                                [--enable_data_sink ENABLE_DATA_SINK] [--data_sink_steps N]
                                [--save_ckpt_path SAVE_CKPT_PATH]
                                [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                                [--save_checkpoint_step N] [--max_ckpt_num N]
                                [--data_dir DATA_DIR] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE] [train_steps N]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --distribute               pre_training by serveral devices: "true"(training by more than 1 device) | "false", default is "false"
    --epoch_size               epoch size: N, default is 1
    --device_id                device id: N, default is 0
    --device_num               number of used devices: N, default is 1
    --save_ckpt_path           path to save checkpoint files: PATH, default is ""
    --max_ckpt_num             max number for saving checkpoint files: N, default is 1
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --data_sink_steps          set data sink steps: N, default is 1
    --save_checkpoint_step     steps for saving checkpoint files: N, default is 1000
    --load_teacher_ckpt_path   path to load teacher checkpoint files: PATH, default is ""
    --data_dir                 path to dataset directory: PATH, default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
    --dataset_type             the dataset type which can be tfrecord/mindrecord, default is tfrecord
```

### Task Distill

```
usage: run_general_task.py  [--device_target DEVICE_TARGET] [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                            [--td_phase1_epoch_size N] [--td_phase2_epoch_size N]
                            [--device_id N] [--do_shuffle DO_SHUFFLE]
                            [--enable_data_sink ENABLE_DATA_SINK] [--save_ckpt_step N]
                            [--max_ckpt_num N] [--data_sink_steps N]
                            [--load_teacher_ckpt_path LOAD_TEACHER_CKPT_PATH]
                            [--load_gd_ckpt_path LOAD_GD_CKPT_PATH]
                            [--load_td1_ckpt_path LOAD_TD1_CKPT_PATH]
                            [--train_data_dir TRAIN_DATA_DIR]
                            [--eval_data_dir EVAL_DATA_DIR]
                            [--task_name TASK_NAME] [--schema_dir SCHEMA_DIR] [--dataset_type DATASET_TYPE]

options:
    --device_target            device where the code will be implemented: "Ascend" | "GPU", default is "Ascend"
    --do_train                 enable train task: "true" | "false", default is "true"
    --do_eval                  enable eval task: "true" | "false", default is "true"
    --td_phase1_epoch_size     epoch size for td phase1: N, default is 10
    --td_phase2_epoch_size     epoch size for td phase2: N, default is 3
    --device_id                device id: N, default is 0
    --do_shuffle               enable shuffle: "true" | "false", default is "true"
    --enable_data_sink         enable data sink: "true" | "false", default is "true"
    --save_ckpt_step           steps for saving checkpoint files: N, default is 1000
    --max_ckpt_num             max number for saving checkpoint files: N, default is 1
    --data_sink_steps          set data sink steps: N, default is 1
    --load_teacher_ckpt_path   path to load teacher checkpoint files: PATH, default is ""
    --load_gd_ckpt_path        path to load checkpoint files which produced by general distill: PATH, default is ""
    --load_td1_ckpt_path       path to load checkpoint files which produced by task distill phase 1: PATH, default is ""
    --train_data_dir           path to train dataset directory: PATH, default is ""
    --eval_data_dir            path to eval dataset directory: PATH, default is ""
    --task_name                classification task: "SST-2" | "QNLI" | "MNLI", default is ""
    --schema_dir               path to schema.json file, PATH, default is ""
    --dataset_type             the dataset type which can be tfrecord/mindrecord, default is tfrecord
```

## Options and Parameters

`gd_config.py` and `td_config.py` contain parameters of BERT model and options for optimizer and lossscale.

In run_distributed_gd_ascend.sh, sink mode is on for default, which is `--enable_data_sink="true"`, each time 100 steps is sinked. For this, epochs from losscallback is the actually count, instead of actual epoch, but `--epoch_size=$EPOCH_SIZE` is still effective.

### Options:

```
batch_size                          batch size of input dataset: N, default is 16
Parameters for lossscale:
    loss_scale_value                initial value of loss scale: N, default is 2^8
    scale_factor                    factor used to update loss scale: N, default is 2
    scale_window                    steps for once updatation of loss scale: N, default is 50

Parameters for optimizer:
    learning_rate                   value of learning rate: Q
    end_learning_rate               value of end learning rate: Q, must be positive
    power                           power: Q
    weight_decay                    weight decay: Q
    eps                             term added to the denominator to improve numerical stability: Q
```

### Parameters:

```
Parameters for bert network:
    seq_length                      length of input sequence: N, default is 128
    vocab_size                      size of each embedding vector: N, must be consistant with the dataset you use. Default is 30522
    hidden_size                     size of bert encoder layers: N
    num_hidden_layers               number of hidden layers: N
    num_attention_heads             number of attention heads: N, default is 12
    intermediate_size               size of intermediate layer: N
    hidden_act                      activation function used: ACTIVATION, default is "gelu"
    hidden_dropout_prob             dropout probability for BertOutput: Q
    attention_probs_dropout_prob    dropout probability for BertAttention: Q
    max_position_embeddings         maximum length of sequences: N, default is 512
    save_ckpt_step                  number for saving checkponit: N, default is 100
    max_ckpt_num                    maximum number for saving checkpoint: N, default is 1
    type_vocab_size                 size of token type vocab: N, default is 2
    initializer_range               initialization value of TruncatedNormal: Q, default is 0.02
    use_relative_positions          use relative positions or not: True | False, default is False
    dtype                           data type of input: mstype.float16 | mstype.float32, default is mstype.float32
    compute_type                    compute type in BertTransformer: mstype.float16 | mstype.float32, default is mstype.float16
```

## [Training Process](#contents)

### Notice

#### Weight transforming of teacher

You have to tranfrom in respond to the original version of bert-base, detailed scripts see https://gitee.com/mindspore/mindspore/commit/52f2d581. **The tranforming script is in tools/tf2ms_ckpt, quick guide included.**

1. config in the code is a 24-layer bert-large, delete the corresponding layer-12~layer-23 and then perform transforming.
2. When transfering bert-base, the shape of "bert/embeddings/word_embeddings" is reversed after the trasfroming between Chinese and English. So if you transfer thew weights of English version of bert-base, you need to change code around line 95 of ms_and_tf_checkpoint_transfer_tools.py to the following code, to handle with the shape of this layer.
   ```
   # for reference
   if len(ms_shape) == 2:
       if ms_shape != tf_shape or ms_shape[0] == ms_shape[1]:
           if(tf_name=="bert/embeddings/word_embeddings"):
               data = tf.transpose(data, (0, 1))
           else:
               data = tf.transpose(data, (1, 0))
           data = data.eval(session=session)
   ```
3. the defalut vocab_size in repo is 30522, in respond to the case insensitive vocab of bert-base, if you use a case sensitive version, vocab_size is 28996(Chinese is 21128), if different versions are used during processing, you need to change the tinybert\src\*\_config.py in repo.

   ```
   # for reference
   BertConfig(
       seq_length=128,
       vocab_size=30522,
       hidden_size=768,
       num_hidden_layers=12,
       num_attention_heads=12,
       intermediate_size=3072,
       hidden_act="gelu",
       hidden_dropout_prob=0.1,
       attention_probs_dropout_prob=0.1,
       max_position_embeddings=512,
       type_vocab_size=2,
       initializer_range=0.02,
       use_relative_positions=False,
       dtype=mstype.float32,
       compute_type=mstype.float16
   )
   ```

   The vocab_size above should be set to you corresponding vocab_size.

4. Since the weights published by Google are trained on GPU, if you use it on NPU, the original loss would be fairly big. We suggest that to load bert_base on NPU and then finetune on enwiki for another time, use it after the loss has converged.

#### The difference of training of task distill

During taks distill phase, we also need teacher. This teacher's weights are not from GD phase, instead, it is from the finetune of GD phase on the task dataset. During the SST-2 task, the teacher from task distill phase is acutally weights from gd, using model from run_classifier.sh (https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/nlp/bert), modify corresponding dataset to finetune. After finetuning for about 40 epochs, and they use it as teacher in the task distill phase of tinybert.

#### Places to be modified when finetuning bert model

The main place is vocab_size, the location is in ./src/finetune_eval_config.py.

### Training

#### running on Ascend

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` and `schma_dir` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```
bash scripts/run_standalone_gd.sh
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```
# grep "epoch" log.txt
epoch: 1, step: 100, outpus are (Tensor(shape=[1], dtype=Float32, 28.2093), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
epoch: 2, step: 200, outpus are (Tensor(shape=[1], dtype=Float32, 30.1724), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

> **Attention** This will bind the processor cores according to the `device_num` and total processor numbers. If you don't expect to run pretraining with binding processor cores, remove the operations about `taskset` in `scripts/run_distributed_gd_ascend.sh`

#### running on GPU

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` `schma_dir` and `device_target=GPU` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```
bash scripts/run_standalone_gd.sh
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```
# grep "epoch" log.txt
epoch: 1, step: 100, outpus are 28.2093
...
```

### Distributed Training

#### running on Ascend

Before running the command below, please check `load_teacher_ckpt_path`, `data_dir` and `schma_dir` has been set. Please set the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```
bash scripts/run_distributed_gd_ascend.sh 8 1 /path/hccl.json
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the LOG\* folder by default. The loss value will be achieved as follows:

```
# grep "epoch" LOG*/log.txt
epoch: 1, step: 100, outpus are (Tensor(shape=[1], dtype=Float32, 28.1478), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
epoch: 1, step: 100, outpus are (Tensor(shape=[1], dtype=Float32, 30.5901), Tensor(shape=[], dtype=Bool, False), Tensor(shape=[], dtype=Float32, 65536))
...
```

#### running on GPU

Please input the path to be the absolute full path, e.g:"/username/checkpoint_100_300.ckpt".

```
bash scripts/run_distributed_gd_gpu.sh 8 1 /path/data/ /path/schema.json /path/teacher.ckpt
```

The command above will run in the background, you can view the results the file log.txt. After training, you will get some checkpoint files under the LOG\* folder by default. The loss value will be achieved as follows:

```
# grep "epoch" LOG*/log.txt
epoch: 1, step: 1, outpus are 63.4098
...
```

## [Evaluation Process](#contents)

### Evaluation

If you want to after running and continue to eval, please set `do_train=true` and `do_eval=true`, If you want to run eval alone, please set `do_train=false` and `do_eval=true`. If running on GPU, please set `device_target=GPU`.

#### evaluation on SST-2 dataset

```
bash scripts/run_standalone_td.sh
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```bash
# grep "The best acc" log.txt
The best acc is 0.872685
The best acc is 0.893515
The best acc is 0.899305
...
The best acc is 0.902777
...
```

#### evaluation on MNLI dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```
bash scripts/run_standalone_td.sh
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```
# grep "The best acc" log.txt
The best acc is 0.803206
The best acc is 0.803308
The best acc is 0.810355
...
The best acc is 0.813929
...
```

#### evaluation on QNLI dataset

Before running the command below, please check the load pretrain checkpoint path has been set. Please set the checkpoint path to be the absolute full path, e.g:"/username/pretrain/checkpoint_100_300.ckpt".

```
bash scripts/run_standalone_td.sh
```

The command above will run in the background, you can view the results the file log.txt. The accuracy of the test dataset will be as follows:

```
# grep "The best acc" log.txt
The best acc is 0.870772
The best acc is 0.871691
The best acc is 0.875183
...
The best acc is 0.891176
...
```

## [Model Description](#contents)

## [Performance](#contents)

### training Performance

| Parameters                  | Ascend                                        | GPU                                                |
| --------------------------- | --------------------------------------------- | -------------------------------------------------- |
| Model Version               | TinyBERT                                      | TinyBERT                                           |
| Resource                    | Ascend 910, cpu:2.60GHz 192cores, memory:755G | NV SMX2 V100-32G, cpu:2.10GHz 64cores, memory:251G |
| uploaded Date               | 08/20/2020                                    | 08/24/2020                                         |
| MindSpore Version           | 1.1.1                                         | 1.1.1                                              |
| Dataset                     | enwiki-128                                    | enwiki-128                                         |
| Training Parameters         | src/gd_config.py                              | src/gd_config.py                                   |
| Optimizer                   | AdamWeightDecay                               | AdamWeightDecay                                    |
| Loss Function               | SoftmaxCrossEntropy                           | SoftmaxCrossEntropy                                |
| outputs                     | probability                                   | probability                                        |
| Loss                        | 6.541583                                      | 6.6915                                             |
| Speed                       | 35.4ms/step                                   | 98.654ms/step                                      |
| Total time                  | 17.3h(20poch, 8p)                              | 48h(20poch, 8p)                                     |
| Params (M)                  | 15M                                           | 15M                                                |
| Checkpoint for task distill | 74M(.ckpt file)                               | 74M(.ckpt file)                                    |

#### Inference Performance

| Parameters          | Ascend          | GPU              |
| ------------------- | --------------- | ---------------- |
| Model Version       |                 |                  |
| Resource            | Ascend 910      | NV SMX2 V100-32G |
| uploaded Date       | 08/20/2020      | 08/24/2020       |
| MindSpore Version   | 1.1.1           | 1.1.1            |
| Dataset             | SST-2,          | SST-2            |
| batch_size          | 32              | 32               |
| Accuracy            | 0.902777        | 0.9086           |
| Speed               |                 |                  |
| Total time          |                 |                  |
| Model for inference | 74M(.ckpt file) | 74M(.ckpt file)  |

### TinyBert Eval in device

|               | AUC      |
| ------------- | -------- |
| on A910       | 0.9039   |
| on A310       | 0.90367  |
| A310 vs. A910 | - 0.03 % |

A310 inference sentence/sec: 31.10


# [Description of Random Situation](#contents)

In run_standaloned_td.sh, we set do_shuffle to shuffle the dataset.

In gd_config.py and td_config.py, we set the hidden_dropout_prob and attention_pros_dropout_prob to dropout some network node.

In run_general_distill.py, we set the random seed to make sure distribute training has the same init weight.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
