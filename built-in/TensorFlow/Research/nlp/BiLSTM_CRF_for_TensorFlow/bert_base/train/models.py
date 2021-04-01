
'\n 一些公共模型代码\n @Time    : 2019/1/30 12:46\n @Author  : MaCan (ma_cancan@163.com)\n @File    : models.py\n'
from npu_bridge.npu_init import *
from bert_base.train.lstm_crf_layer import BLSTM_CRF
from tensorflow.contrib.layers.python.layers import initializers
__all__ = ['InputExample', 'InputFeatures', 'decode_labels', 'create_model', 'convert_id_str', 'convert_id_to_label', 'result_to_json', 'create_classification_model']

class Model(object):

    def __init__(self, *args, **kwargs):
        pass

class InputExample(object):
    'A single training/test example for simple sequence classification.'

    def __init__(self, guid=None, text=None, label=None):
        'Constructs a InputExample.\n        Args:\n          guid: Unique id for the example.\n          text_a: string. The untokenized text of the first sequence. For single\n            sequence tasks, only this sequence must be specified.\n          label: (Optional) string. The label of the example. This should be\n            specified for train and dev examples, but not for test examples.\n        '
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    'A single set of features of data.'

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class DataProcessor(object):
    'Base class for data converters for sequence classification data sets.'

    def get_train_examples(self, data_dir):
        'Gets a collection of `InputExample`s for the train set.'
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        'Gets a collection of `InputExample`s for the dev set.'
        raise NotImplementedError()

    def get_labels(self):
        'Gets the list of labels for this data set.'
        raise NotImplementedError()

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings, dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    '\n    创建X模型\n    :param bert_config: bert 配置\n    :param is_training:\n    :param input_ids: 数据的idx 表示\n    :param input_mask:\n    :param segment_ids:\n    :param labels: 标签的idx 表示\n    :param num_labels: 类别数量\n    :param use_one_hot_embeddings:\n    :return:\n    '
    import tensorflow as tf
    from bert_base.bert import modeling
    model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=use_one_hot_embeddings)
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers, dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels, seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return rst

def create_classification_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    '\n\n    :param bert_config:\n    :param is_training:\n    :param input_ids:\n    :param input_mask:\n    :param segment_ids:\n    :param labels:\n    :param num_labels:\n    :param use_one_hot_embedding:\n    :return:\n    '
    import tensorflow as tf
    from bert_base.bert import modeling
    model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids)
    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[(- 1)].value
    output_weights = tf.get_variable('output_weights', [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable('output_bias', [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope('loss'):
        if is_training:
            output_layer = npu_ops.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=(- 1))
        log_probs = tf.nn.log_softmax(logits, axis=(- 1))
        if (labels is not None):
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = (- tf.reduce_sum((one_hot_labels * log_probs), axis=(- 1)))
            loss = tf.reduce_mean(per_example_loss)
        else:
            (loss, per_example_loss) = (None, None)
    return (loss, per_example_loss, logits, probabilities)

def decode_labels(labels, batch_size):
    new_labels = []
    for row in range(batch_size):
        label = []
        for i in labels[row]:
            i = i.decode('utf-8')
            if (i == '**PAD**'):
                break
            if (i in ['[CLS]', '[SEP]']):
                continue
            label.append(i)
        new_labels.append(label)
    return new_labels

def convert_id_str(input_ids, batch_size):
    res = []
    for row in range(batch_size):
        line = []
        for i in input_ids[row]:
            i = i.decode('utf-8')
            if (i == '**PAD**'):
                break
            if (i in ['[CLS]', '[SEP]']):
                continue
            line.append(i)
        res.append(line)
    return res

def convert_id_to_label(pred_ids_result, idx2label, batch_size):
    '\n    将id形式的结果转化为真实序列结果\n    :param pred_ids_result:\n    :param idx2label:\n    :return:\n    '
    result = []
    index_result = []
    for row in range(batch_size):
        curr_seq = []
        curr_idx = []
        ids = pred_ids_result[row]
        for (idx, id) in enumerate(ids):
            if (id == 0):
                break
            curr_label = idx2label[id]
            if (curr_label in ['[CLS]', '[SEP]']):
                if ((id == 102) and ((idx < len(ids)) and (ids[(idx + 1)] == 0))):
                    break
                continue
            curr_seq.append(curr_label)
            curr_idx.append(id)
        result.append(curr_seq)
        index_result.append(curr_idx)
    return (result, index_result)

def result_to_json(self, string, tags):
    '\n    将模型标注序列和输入序列结合 转化为结果\n    :param string: 输入序列\n    :param tags: 标注结果\n    :return:\n    '
    item = {'entities': []}
    entity_name = ''
    entity_start = 0
    idx = 0
    last_tag = ''
    for (char, tag) in zip(string, tags):
        if (tag[0] == 'S'):
            self.append(char, idx, (idx + 1), tag[2:])
            item['entities'].append({'word': char, 'start': idx, 'end': (idx + 1), 'type': tag[2:]})
        elif (tag[0] == 'B'):
            if (entity_name != ''):
                self.append(entity_name, entity_start, idx, last_tag[2:])
                item['entities'].append({'word': entity_name, 'start': entity_start, 'end': idx, 'type': last_tag[2:]})
                entity_name = ''
            entity_name += char
            entity_start = idx
        elif (tag[0] == 'I'):
            entity_name += char
        elif (tag[0] == 'O'):
            if (entity_name != ''):
                self.append(entity_name, entity_start, idx, last_tag[2:])
                item['entities'].append({'word': entity_name, 'start': entity_start, 'end': idx, 'type': last_tag[2:]})
                entity_name = ''
        else:
            entity_name = ''
        entity_start = idx
        idx += 1
        last_tag = tag
    if (entity_name != ''):
        self.append(entity_name, entity_start, idx, last_tag[2:])
        item['entities'].append({'word': entity_name, 'start': entity_start, 'end': idx, 'type': last_tag[2:]})
    return item
