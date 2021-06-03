# 训练数据集 enwiki 处理

1 enwiki 提取时候尽量切分的更小，防止内存溢出，使用脚本挨个处理，否则内存占用大，且速度慢

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

# SST-2 数据集处理

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

# 训练 teacher 权重转换

需要对原版 bert-base 的权重进行转换，具体转换脚本可以参考 https://gitee.com/mindspore/mindspore/commit/52f2d581  
注意

1. 这份转换代码中的 config 是 24 层的 bert-large,删除对应 layer12~layer23 后，才能转换 bert-base
2. 转换 bert-base，中文和英文转换后其中“bert/embeddings/word_embeddings”这一层的 shape 是反的，所以如果转化英文 bert-base 的权重需要将 ms_and_tf_checkpoint_transfer_tools.py 中的 95 行附近需要改成如下代码，特殊处理这一层的 shape。

```
if len(ms_shape) == 2:
    if ms_shape != tf_shape or ms_shape[0] == ms_shape[1]:
        if(tf_name=="bert/embeddings/word_embeddings"):
            data = tf.transpose(data, (0, 1))
        else:
            data = tf.transpose(data, (1, 0))
        data = data.eval(session=session)
```

3. repo 中默认的 vocab_size 是 30522,对标 uncased 的 bert-base 的,使用的如果是 cased，vocab_size 是 28996(中文为 21128)，使用过程中如果使用的是不同的版本，需要修改 repo 中的 tinybert\src\*\_config.py 文件，

```
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

如上 config 代码中的 vocab_size 设置为你使用的版本对应的 vocab_size。  
4. 由于谷歌发布的权重是在 gpu 上预训练收敛的，在 NPU 上使用初始 loss 还是比较大，建议 bert_base 在 NPU 上加载后再 enwiki 上 finetune 一次，把 loss 收敛后再放到后面 GD 步骤使用

# 数据下沉计算的说明

并行计算脚本 run_distributed_gd_ascend.sh 中，默认开启了数据下沉，即--enable_data_sink="true"，每次下沉 100 个 step。这时 losscallback 输出的 epoch 为数据下沉的次数，而非实际的 epoch，但是实际--epoch_size=$EPOCH_SIZE 参数还是生效的。

# dataset 的 schema.json 的说明

src\dataset.py 中可以看到 53 行，

```
ds = de.TFRecordDataset(data_files, schema_dir, columns_list=columns_list,
```

如果没有 schema.json 文件，建议将以上 schema_dir 变量直接改为 None。否则运行会由于该变量没有定以而报错。  
这个文件的主要作用是定以数据载入的限制部分，一般没有可以直接设为 None。

# TD 阶段训练的区别

TD 训练阶段，也需要 teacher，这个 teacher 不是 GD 阶段的权重，是 GD 预训练权重在 task 的数据集上 finetune 之后的权重，在本次 SST-2 任务中，td 阶段的 teacher 实际为 gd 阶段的权重，使用 bert 模型 https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/nlp/bert 中的 run_classifier.sh 脚本，修改对应数据集进行 finetune，finetune 40 个 Epoch,然后再作为 tinybert 的 TD 阶段的 teacher 权重。

# bert 模型 finetune 需要修改的地方

主要修改的地方是 vocab_size，finetune 修改的位置在./src/finetune_eval_config.py

#

```
docker run  -v /mnt/backup/tinybert/tfrecord/:/home/admin/dataset/enwiki/tfrecord/ -v /mnt/backup/tinybert/SST-2/:/home/admin/dataset/SST-2/ -v /mnt/backup/tinybert/tinybert/:/home/admin/code/tinybert/ -v /usr/local/Ascend:/usr/local/Ascend -v /home/HwHiAiUser/Ascend/:/home/HwHiAiUser/Ascend/ --privileged -it f12904d19b16 bash

docker run  -v /ceph_mount/pvc/aiplatform-app-data/work/admin:/home/admin -v /usr/local/Ascend:/usr/local/Ascend -v /home/HwHiAiUser/Ascend/:/home/HwHiAiUser/Ascend/ --privileged -it f12904d19b16 bash

python export.py --ckpt_file=/home/admin/code/tinybert/scripts/LOG1/2021-03-05_time_08_24_46/tiny_bert_15_10000.ckpt

python create_pretraining_data.py --input_file=./enwiki/extracted/AA/wiki_15 --output_file=./enwiki/step/enwiki_15.tfrecord --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5

```
