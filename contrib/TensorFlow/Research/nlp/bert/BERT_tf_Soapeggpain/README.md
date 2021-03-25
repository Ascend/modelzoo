# Bert-NV-NPU Finetune SQuAD v2.0阅读理解训练

本脚本实现了Bert-NV-NPU Finetune SQuAD v2.0阅读理解训练迁移至Ascend910的单p及多p训练，精度超过NVIDIA精度值

## 1. GPU源码地址

[NVIDIA/DeepLearningExamples/TensorFlow/LanguageModeling/BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

## 2. 精度结果

[SQuAD v2.0数据集](https://rajpurkar.github.io/SQuAD-explorer/)，采用Google发布的[Bert-Base 12层Uncase Pretrain模型(vocab_size=30522)](https://github.com/google-research/bert)进行Finetune

主要训练超参max_seq_len=384，batch_size=32，epochs=2，learning_rate=5e-5

在Ascend910 单p环境，使用混合精度模式训练(环境版本：1.75.T15.200.B150)

结果如下：

exact_match: 73.38499115640529

f1: 76.51607891633849

超过了NVIDIA在NGC上发布的Bert-Base-Finetune SQuAD2.0 fp32的精度(em: 72.93017771414132, f1: 76.26978130334918)，见[NGC-BERT-Base(fine-tuning) - SQuAD 2.0](https://ngc.nvidia.com/catalog/models/nvidia:bert_tf_v2_base_fp32_384/files)

[NPU训练的ckpt获取](https://pan.baidu.com/s/10Nbt4YUI7Pg2vOqEqmtwTQ)，提取码：zkar

其中包括了NPU训练结果的ckpt模型，推理的结果：predictions.json

Nvidia训练日志见log/gpu下文件：tf_bert_squad_1n_fp32_gbs32.190523203805.log

## 3. 文件说明

- code: 该目录下为Bert-Base SQuAD迁移至NPU的训练脚本

- dataset: 放有SQuAD v2.0数据集及评估脚本

- log: gpu为NVIDIA提供的Bert-Base-Finetune SQuAD2.0 fp32训练日志；npu为Ascend910上训练日志

- script: 该目录下放有e2e脚本，方便在NPU上的一键执行训练

- casecsv: e2e脚本所用的测试用例文件


## 4. 执行方法

- 完整训练+推理+评估：

    命令行操作：

    配置好环境变量后，将预训练模型，数据集等放至指定路径，执行下列命令：

    ```shell
    python3.7 run_squad.py --vocab_file=/data/ckpt/bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=/data/ckpt/bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data/ckpt/bert/uncased_L-12_H-768_A-12/bert_model.ckpt --do_train=True --train_file=/data/dataset/bert/finetune/squad_v2.0/train-v2.0.json --input_files_dir=/data/dataset/bert/finetune/squad_v2.0/tfrecord/train --use_tfrecord=False --do_predict=True --predict_file=/data/dataset/bert/finetune/squad_v2.0/dev-v2.0.json --eval_script=/data/dataset/bert/finetune/squad_v2.0/evaluate-v2.0.py --train_batch_size=32 --learning_rate=5.6e-5 --num_train_epochs=2 --max_seq_length=384 --doc_stride=128 --amp=True --npu_bert_debug=False --npu_bert_use_tdt=True --iterations_per_loop=100 --save_checkpoints_steps=1000 --npu_bert_clip_by_global_norm=True --distributed=False --npu_bert_loss_scale=0 --version_2_with_negative=True --predict_batch_size=1 --output_dir=./result
    ```

    e2e脚本：

    将code，dataset，csv放至e2e脚本对应路径下面，执行下列命令：

    ```shell
    bash e2e_test.sh casecsv/case_bert_nv_finetune_squad.csv 2 host
    ```

- 仅推理+评估：

    配置好环境变量后，将训练完成的模型，数据集等放至指定路径，执行下列命令：

    ```shell
    python3.7 run_squad.py --vocab_file=/data/ckpt/bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=/data/ckpt/bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data/ckpt/bert/base_squad_v2.0/model.ckpt-8200 --do_train=False --train_file=/data/dataset/bert/finetune/squad_v2.0/train-v2.0.json --input_files_dir=/data/dataset/bert/finetune/squad_v2.0/tfrecord/train --use_tfrecord=True --do_predict=True --predict_file=/data/dataset/bert/finetune/squad_v2.0/dev-v2.0.json --eval_script=/data/dataset/bert/finetune/squad_v2.0/evaluate-v2.0.py --train_batch_size=32 --learning_rate=5.6e-5 --num_train_epochs=2 --max_seq_length=384 --doc_stride=128 --amp=True --npu_bert_debug=False --npu_bert_use_tdt=True --iterations_per_loop=100 --save_checkpoints_steps=1000 --npu_bert_clip_by_global_norm=True --distributed=False --npu_bert_loss_scale=0 --version_2_with_negative=True --predict_batch_size=1 --output_dir=./result/
    ```

    e2e脚本：

    将code，dataset，csv放至e2e脚本对应路径下面，执行下列命令：
    ```shell
    bash e2e_test.sh casecsv/case_bert_nv_finetune_squad.csv 3 host
    ```

- 仅评估：

    ```shell
    python3.7 /data/dataset/bert/finetune/squad_v2.0/evaluate-v2.0.py /data/dataset/bert/finetune/squad_v2.0/dev-v2.0.json predictions.json
    ```