# TinyBERT 概述

从推理角度看，[TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)比[BERT-base](https://github.com/google-research/bert)（BERT 模型基础版本）体积小了 7.5 倍、速度快了 9.4 倍，自然语言理解的性能表现更突出。TinyBert 在预训练和任务学习两个阶段创新采用了转换蒸馏。

[论文](https://arxiv.org/abs/1909.10351): Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351). arXiv preprint arXiv:1909.10351.

# 模型架构

TinyBERT 模型的主干结构是转换器，转换器包含四个编码器模块，其中一个为自注意模块。一个自注意模块即为一个注意模块。

# 数据集

- 下载 zhwiki 或 enwiki 数据集进行一般蒸馏。使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本。如需将数据集转化为 TFRecord 格式。详见[BERT](https://github.com/google-research/bert)代码库中的 create_pretraining_data.py 文件。

  - enwiki 提取时候尽量切分的更小，防止内存溢出，使用脚本挨个处理，否则内存占用大，且速度慢

    ```
    git clone https://github.com/attardi/wikiextractor
    python WikiExtractor.py -b 128M -o ../../extracted ../../enwiki-latest-pages-articles.xml.bz2
    ```

    以下指令中的 vocab_file 为在 google 的 bertrepo 中下载对应版本 bert-base 的 checkpoint 内对应的文件。

    ```
    sudo pip install bert-tensorflow
    python create_pretraining_data.py --input_file=./enwiki/extracted/AA/wiki_00 --output_file=./enwiki/tfrecord/enwiki_00.tfrecord --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5
    ```

    由于切分后文件多，且处理时间较长，合并处理容易出现内存问题，建议挨个处理转换，转换脚本可以参考如下的 shell 代码

    ```
    for i in {17..24}
    do
        python create_pretraining_data.py --input_file=./enwiki/extracted/AA/wiki_${i} --output_file=./enwiki/tfrecord/enwiki_${i}.tfrecord --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5  &
    done
    ```

    由于转换脚本是单线程执行，如果在服务器上，且内存够大，可以选择启动多个脚本，将数据集拆分并行转换，主要修改的地方就在于上面代码的 for i in {17..24}部分，修改数字为该线程处理的部分就可以了

- 下载 GLUE 数据集进行任务蒸馏。将数据集由 JSON 格式转化为 TFRecord 格式。详见[BERT](https://github.com/google-research/bert)代码库中的 run_classifier.py 文件。

  - SST-2 数据集处理  
     该数据集为 glue benchmark 中的标准数据集，地址为：https://dl.fbaipublicfiles.com/glue/data/SST-2.zip  
     在 run_classifier.py 文件中补充如下代码。

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

    然后使用如下指令

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

## 最终数据集目录

1. `├─dataset`
2. `│ ├─SST-2`
3. `│ │ ├─eval`
4. `│ │ | ├─eval.tf_record`
5. `│ │ ├─train`
6. `│ │ | ├─train.tf_record`
7. `│ |—wiki`
8. `│ │ ├─enwiki_00.tfrecord`
9. `│…`

# 环境要求

- 硬件（Ascend 或 GPU）
  - 使用 Ascend 或 GPU 处理器准备硬件环境。
- 框架
  - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于 Mindspore 的信息，请查看以下资源：
  - [MindSpore 教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

# 训练

## 订阅算法

订阅算法流程请参考[使用 AI 市场的预置算法训练模型-订阅算法](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0025.html#modelarts_10_0025__section87421022184315)。

## 创建训练作业

数据预处理完成之后，将预处理后的数据迁移至 OBS，上传 OBS 请参考[如何上传数据至 OBS？](https://support.huaweicloud.com/modelarts_faq/modelarts_05_0013.html) 。

## 训练参数说明

| 名称                     | 默认值          | 类型   | 是否必填 | 描述                                                              |
| ------------------------ | --------------- | ------ | -------- | ----------------------------------------------------------------- |
| distill_type             | td              | string | 是       | 选择通用蒸馏或任务蒸馏[td,gd]                                     |
| device_target            | Ascend          | string | 是       | 硬件选择[Ascend,GPU]                                              |
| distribute               | true            | string | 是       | 是否分布式[true,false]                                            |
| td_do_train              | true            | string | 否       | 是否进行通用蒸馏训练[true,false]                                  |
| td_do_eval               | true            | string | 否       | 是否进行通用蒸馏评估[true,false]                                  |
| td_phase1_epoch_size     | 10              | int    | 否       | 在对任务数据集 finetune 时的断点文件                              |
| td_task_name             | SST-2           | string | 否       | 训练集名称[SST-2,QNLI,MNLI]                                       |
| td_phase2_epoch_size     | 3               | int    | 否       | 通用蒸馏第二阶段训练轮数                                          |
| td_load_teacher_ckpt_obs | ""              | String | 否       | 在对任务数据集 finetune 时的断点文件                              |
| td_load_gd_ckpt_obs      | ""              | string | 否       | 任务蒸馏步骤：把在任务蒸馏 ckpt 文件加载入 tinybert，用于最后评估 |
| td_load_td1_ckpt_obs     | ""              | string | 否       | 任务蒸馏步骤：把在任务蒸馏 ckpt 文件加载入 tinybert，用于最后评估 |
| gd_epoch_size            | 3               | int    | 否       | 通用蒸馏轮数                                                      |
| gd_device_num            | 1               | int    | 否       | 通用蒸馏用设备数量                                                |
| gd_save_ckpt_path        | /tmp/ckpt_save/ | string | 否       | 保存 ckpt 的路径                                                  |
| gd_resume_ckpt_obs       | ""              | string | 否       | 通用蒸馏断点加载文件                                              |
| gd_load_teacher_ckpt_obs | ""              | string | 否       | 通用蒸馏老师的断点加载文件                                        |
| enable_data_sink         | true            | string | 否       | 是否使用数据下沉模式[true,false]                                  |
| data_sink_steps          | 1               | int    | 否       | 通用蒸馏中使用，每轮的 sink step                                  |
| save_ckpt_step           | 100             | int    | 否       | 多少步保存 ckpt                                                   |
| max_ckpt_num             | 1               | int    | 否       | 最多保存的 ckpt 数量                                              |
| dataset_type             | tfrecord        | string | 否       | 数据集类型[tfrecord,mindrecord,tfrecord]                          |
| export_file_format       | AIR             | string | 否       | 转换文件类型[AIR,ONNX,MINDIR]                                     |
| file_name                | tinybert        | string | 否       | 输出数据名                                                        |
| do_shuffle               | true            | string | 否       | 是否 shuffle[true,false]                                          |

## 通用蒸馏(General Distill)需要参数

若运行通用蒸馏，下面这些参数是必要的，且应设`distill_type=gd`.

| 名称                     | 默认值          | 类型   | 是否必填 | 描述                          |
| ------------------------ | --------------- | ------ | -------- | ----------------------------- |
| distill_type             | td              | string | 是       | 选择通用蒸馏或任务蒸馏[td,gd] |
| device_target            | Ascend          | string | 是       | 硬件选择[Ascend,GPU]          |
| distribute               | true            | string | 是       | 是否分布式[true,false]        |
| gd_epoch_size            | 3               | int    | 否       | 通用蒸馏轮数                  |
| gd_device_num            | 1               | int    | 否       | 通用蒸馏用设备数量            |
| gd_save_ckpt_path        | /tmp/ckpt_save/ | string | 否       | 保存 ckpt 的路径              |
| gd_resume_ckpt_obs       | ""              | string | 否       | 通用蒸馏断点加载文件          |
| gd_load_teacher_ckpt_obs | ""              | string | 否       | 通用蒸馏老师的断点加载文件    |

## 任务蒸馏(Task Distill)需要参数

若运行任务蒸馏，下面这些参数是必要的，且应设`distill_type=td`.

| 名称                     | 默认值 | 类型   | 是否必填 | 描述                                                              |
| ------------------------ | ------ | ------ | -------- | ----------------------------------------------------------------- |
| distill_type             | td     | string | 是       | 选择通用蒸馏或任务蒸馏[td,gd]                                     |
| device_target            | Ascend | string | 是       | 硬件选择[Ascend,GPU]                                              |
| distribute               | true   | string | 是       | 是否分布式[true,false]                                            |
| enable_data_sink         | true   | string | 否       | 是否使用数据下沉模式[true,false]                                  |
| data_sink_steps          | 1      | int    | 否       | 通用蒸馏中使用，每轮的 sink step                                  |
| td_do_train              | true   | string | 否       | 是否进行通用蒸馏训练[true,false]                                  |
| td_do_eval               | true   | string | 否       | 是否进行通用蒸馏评估[true,false]                                  |
| td_phase1_epoch_size     | 10     | int    | 否       | 在对任务数据集 finetune 时的断点文件                              |
| td_task_name             | SST-2  | string | 否       | 训练集名称[SST-2,QNLI,MNLI]                                       |
| td_phase2_epoch_size     | 3      | int    | 否       | 通用蒸馏第二阶段训练轮数                                          |
| td_load_teacher_ckpt_obs | ""     | String | 否       | 在对任务数据集 finetune 时的断点文件                              |
| td_load_gd_ckpt_obs      | ""     | string | 否       | 任务蒸馏步骤：把在任务蒸馏 ckpt 文件加载入 tinybert，用于最后评估 |
| td_load_td1_ckpt_obs     | ""     | string | 否       | 任务蒸馏步骤：把在任务蒸馏 ckpt 文件加载入 tinybert，用于最后评估 |
