
'\n基于命令行的在线预测方法\n@Author: Macan (ma_cancan@163.com) \n'
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
from datetime import datetime
from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
args = get_args_parser()
model_dir = 'C:\\Users\\C\\Documents\\Tencent Files\\389631699\\FileRecv\\semi_corpus_people_2014'
bert_dir = 'F:\\chinese_L-12_H-768_A-12'
is_training = False
use_one_hot_embeddings = False
batch_size = 1
gpu_config = npu_config_proto(config_proto=tf.ConfigProto())
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=npu_session_config_init(session_config=gpu_config))
model = None
global graph
(input_ids_p, input_mask_p, label_ids_p, segment_ids_p) = (None, None, None, None)
print('checkpoint path:{}'.format(os.path.join(model_dir, 'checkpoint')))
if (not os.path.exists(os.path.join(model_dir, 'checkpoint'))):
    raise Exception('failed to get checkpoint. going to return ')
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for (key, value) in label2id.items()}
with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = (len(label_list) + 1)
graph = tf.get_default_graph()
with graph.as_default():
    print('going to restore checkpoint')
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name='input_ids')
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name='input_mask')
    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None, labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)

def predict_online():
    '\n    do online prediction. each time make prediction for one instance.\n    you can change to a batch if you want.\n\n    :param line: a list. element is: [dummy_label,text_a,text_b]\n    :return:\n    '

    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids], (batch_size, args.max_seq_length))
        return (input_ids, input_mask, segment_ids, label_ids)
    global graph
    with graph.as_default():
        print(id2label)
        while True:
            print('input the test sentence:')
            sentence = str(input())
            start = datetime.now()
            if (len(sentence) < 2):
                print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence)
            (input_ids, input_mask, segment_ids, label_ids) = convert(sentence)
            feed_dict = {input_ids_p: input_ids, input_mask_p: input_mask}
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            print(pred_label_result)
            result = strage_combined_link_org_loc(sentence, pred_label_result[0])
            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))

def convert_id_to_label(pred_ids_result, idx2label):
    '\n    将id形式的结果转化为真实序列结果\n    :param pred_ids_result:\n    :param idx2label:\n    :return:\n    '
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if (ids == 0):
                break
            curr_label = idx2label[ids]
            if (curr_label in ['[CLS]', '[SEP]']):
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result

def strage_combined_link_org_loc(tokens, tags):
    '\n    组合策略\n    :param pred_label_result:\n    :param types:\n    :return:\n    '

    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        print(', '.join(line))
    params = None
    eval = Result(params)
    if (len(tokens) > len(tags)):
        tokens = tokens[:len(tags)]
    (person, loc, org) = eval.get_result(tokens, tags)
    print_output(loc, 'LOC')
    print_output(person, 'PER')
    print_output(org, 'ORG')

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    '\n    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中\n    :param ex_index: index\n    :param example: 一个样本\n    :param label_list: 标签列表\n    :param max_seq_length:\n    :param tokenizer:\n    :param mode:\n    :return:\n    '
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    if (not os.path.exists(os.path.join(model_dir, 'label2id.pkl'))):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    tokens = example
    if (len(tokens) >= (max_seq_length - 1)):
        tokens = tokens[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append('[CLS]')
    segment_ids.append(0)
    label_ids.append(label_map['[CLS]'])
    for (i, token) in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append('[SEP]')
    segment_ids.append(0)
    label_ids.append(label_map['[SEP]'])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = ([1] * len(input_ids))
    while (len(input_ids) < max_seq_length):
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append('**NULL**')
    assert (len(input_ids) == max_seq_length)
    assert (len(input_mask) == max_seq_length)
    assert (len(segment_ids) == max_seq_length)
    assert (len(label_ids) == max_seq_length)
    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
    return feature

class Pair(object):

    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def merge(self):
        return self.__merge

    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types

    @word.setter
    def word(self, word):
        self.__word = word

    @start.setter
    def start(self, start):
        self.__start = start

    @end.setter
    def end(self, end):
        self.__end = end

    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)

class Result(object):

    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []

    def get_result(self, tokens, tags, config=None):
        self.result_to_json(tokens, tags)
        return (self.person, self.loc, self.org)

    def result_to_json(self, string, tags):
        '\n        将模型标注序列和输入序列结合 转化为结果\n        :param string: 输入序列\n        :param tags: 标注结果\n        :return:\n        '
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

    def append(self, word, start, end, tag):
        if (tag == 'LOC'):
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif (tag == 'PER'):
            self.person.append(Pair(word, start, end, 'PER'))
        elif (tag == 'ORG'):
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))
if (__name__ == '__main__'):
    predict_online()
