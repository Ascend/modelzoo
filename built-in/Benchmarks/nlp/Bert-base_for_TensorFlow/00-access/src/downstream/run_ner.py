# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import pickle
import collections
import csv
import os
import time 

import modeling
import optimization
import tokenization
import tensorflow as tf
import tf_metrics

from npu_bridge.estimator.npu.npu_config import *
from npu_bridge.estimator.npu.npu_estimator import *
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_estimator import NPUEstimatorSpec

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'ner', "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval_all", True, "Whether to run eval on the dev set all.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_train_and_eval", False, "train and eval")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 10,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool('npu_bert_debug', False, 'If True, dropout and shuffle is disabled.')

flags.DEFINE_bool('npu_bert_fused_gelu', True, 'Whether to use npu defined gelu op')

flags.DEFINE_bool('npu_bert_npu_dropout', True, 'Whether to use npu defined gelu op')

flags.DEFINE_integer("npu_bert_loss_scale", 0, "Whether to use loss scale, -1 is disable, 0 is dynamic loss scale, >=1 is static loss scale")

flags.DEFINE_integer('init_loss_scale_value', 2**32, 'Initial loss scale value for loss scale optimizer')

flags.DEFINE_bool('hcom_parallel', True, 'Whether to use parallel allreduce')

class _LogSessionRunHook(tf.train.SessionRunHook):
  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=['loss_scale:0'])

  def after_run(self, run_context, run_values):
    loss_scaler = run_values.results
    print("loss_scale:", loss_scaler, flush=True)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        # self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # 句子配对数据读取
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    # 多分类中文数据读取
    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("gbk").split("\t"))
        return lines

    # 序列标注任务数据读取
    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contends) == 0 and words[-1] == '。':
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines

class Ner_processor(DataProcessor):
    '''
    命名实体识别
    文件格式：txt
    数据格式：字 标注
    '''

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples


def write_tokens(tokens, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer,
                           # 序列标注任务write_tokens才有用
                           mode=None):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    ### 序列标注问题数据解析方式
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    # 保存label->index 的map
    if not os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text_a.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过句首和句尾使用不同的标志来标注，使用CLS也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_ids,
    )

    write_tokens(ntokens, mode)
    return feature

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,
        mode=None):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        # features["is_real_example"] = create_int_feature(
        #     [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            if not FLAGS.npu_bert_debug:
                d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    # 使用数据加载BertModel,获取对应的字embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    with tf.variable_scope("ner_classifier"):
        output_layer = model.get_sequence_output()

        hidden_size = output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )
        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            # 将词性标注问题看做是对每个位置的词的分类问题
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 11])
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=-1)

        return (loss, per_example_loss, logits, predict)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 加载的模型参数
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            hooks=[]
            if FLAGS.npu_bert_loss_scale == 0:
                hooks.append(_LogSessionRunHook())
            output_spec = NPUEstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, scaffold=scaffold_fn, training_hooks=hooks)
        elif mode == tf.estimator.ModeKeys.EVAL:
            ### 词性标注问题评价函数设计
            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="micro")
                recall = tf_metrics.recall(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="micro")
                f = tf_metrics.f1(label_ids, predictions, 11, [2, 3, 4, 5, 6, 7], average="micro")
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                }
            eval_metrics = metric_fn(per_example_loss, label_ids, logits)
            output_spec = NPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:   
            output_spec = NPUEstimatorSpec(
                    mode=mode,
                    predictions=predicts,
                    scaffold=scaffold_fn)    
        return output_spec
    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        #batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            if not FLAGS.npu_bert_debug:
                d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    for name, value in FLAGS.__flags.items():
      print("TF flags:", name, "      ", FLAGS[name].value)

    processors = {
        "ner": Ner_processor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = NPURunConfig(
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=0,
        save_summary_steps=0,
        session_config=tf.ConfigProto(),
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        log_step_count_steps=1,
        enable_data_pre_proc=True,
        iterations_per_loop=FLAGS.iterations_per_loop,
        hcom_parallel=FLAGS.hcom_parallel)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train or FLAGS.do_predict:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        tf.logging.info('***********label_list of this task is %s'%str(label_list))
        time.sleep(10)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    ### 对于词性标注问题，注意num_label的值
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = NPUEstimator(
        model_fn=model_fn,
        config=run_config,)

    best_ckpt = ""
    best_item = ""
    if FLAGS.do_train or FLAGS.do_predict:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size)
        if FLAGS.do_train_and_eval:
            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        if FLAGS.do_train:
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval or FLAGS.do_eval_all:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
        
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        
        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.

        eval_drop_remainder = True
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            batch_size=FLAGS.eval_batch_size)
        if FLAGS.do_train_and_eval:
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=1*60, throttle_secs=60)#eval_steps
        elif FLAGS.do_eval_all:
            steps_and_files = []
            eval_all_list = []
            filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
                    global_step = int(cur_filename.split("-")[-1])
                    tf.logging.info("Add {} to eval list.".format(cur_filename))
                    steps_and_files.append([global_step,cur_filename])
            steps_and_files = sorted(steps_and_files, key = lambda x:x[0])
            output_eval_all_file = os.path.join(FLAGS.output_dir, "eval_all_result.txt")
            output_eval_all_file_sorted = os.path.join(FLAGS.output_dir, "eval_all_result_sorted.txt")
            tf.logging.info("output_eval_all_file: "+output_eval_all_file)
            with tf.gfile.GFile(output_eval_all_file, "w") as writer:
                for global_step, filename in sorted(steps_and_files, key = lambda x:x[0]):
                    time.sleep(15)
                    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)
                    eval_all_list.append([result["eval_precision"],global_step,filename])
                    writer.write("*****Eval results %s *****\n" % (filename))
                    for key in sorted(result.keys()):
                        tf.logging.info("%s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
            eval_all_list = sorted(eval_all_list, key=lambda x:x[0], reverse = True)
            if(len(eval_all_list)!=0):
                best_item = str(eval_all_list[0])
                best_ckpt = str(eval_all_list[0][2])
            with open(output_eval_all_file_sorted, "w") as writer:
                for ele in eval_all_list:
                    writer.write(str(ele))
                    writer.write('\n')
        else:
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_train_and_eval:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if FLAGS.do_predict or FLAGS.do_eval_all:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open(os.path.join(FLAGS.output_dir, "label2id.pkl"), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_drop_remainder = True 
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            batch_size=FLAGS.predict_batch_size)
        if FLAGS.do_eval_all:
            FLAGS.init_checkpoint = best_ckpt
        result = estimator.evaluate(input_fn=predict_input_fn,checkpoint_path=FLAGS.init_checkpoint)#predict
        eval_re_path=os.path.join(FLAGS.output_dir, "eval")
        if not os.path.exists(eval_re_path):
            os.mkdir(eval_re_path) 
        output_test_file = os.path.join(eval_re_path, "test_results.txt")
        with tf.gfile.GFile(output_test_file, "w") as writer:
            tf.logging.info("***** Test results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            if FLAGS.do_eval_all:
                tf.logging.info("***** Eval Best results *****")
                tf.logging.info(best_item)
                writer.write(best_item)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
