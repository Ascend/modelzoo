
'\n#@Time    : ${DATE} ${TIME}\n# @Author  : MaCan (ma_cancan@163.com)\n# @File    : ${NAME}.py\n'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import os
import flask
from flask import request, jsonify
import json
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import sys
sys.path.append('../..')
from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
model_dir = '../../output'
bert_dir = 'H:\\models\\chinese_L-12_H-768_A-12'
is_training = False
use_one_hot_embeddings = False
batch_size = 1
max_seq_length = 202
gpu_config = npu_config_proto(config_proto=tf.ConfigProto())
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=npu_session_config_init(session_config=gpu_config))
model = None
global graph
(input_ids_p, input_mask_p, label_ids_p, segment_ids_p) = (None, None, None, None)
print('checkpoint path:{}'.format(os.path.join(model_dir, 'checkpoint')))
if (not os.path.exists(os.path.join(model_dir, 'checkpoint'))):
    raise Exception('failed to get checkpoint. going to return ')
with open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for (key, value) in label2id.items()}
with open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = (len(label_list) + 1)
graph = tf.get_default_graph()
with graph.as_default():
    print('going to restore checkpoint')
    input_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name='input_ids')
    input_mask_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name='input_mask')
    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None, labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=True)
app = flask.Flask(__name__)

@app.route('/ner_predict_service', methods=['GET'])
def ner_predict_service():
    '\n    do online prediction. each time make prediction for one instance.\n    you can change to a batch if you want.\n\n    :param line: a list. element is: [dummy_label,text_a,text_b]\n    :return:\n    '

    def convert(line):
        feature = convert_single_example(0, line, label_list, max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, max_seq_length))
        label_ids = np.reshape([feature.label_ids], (batch_size, max_seq_length))
        return (input_ids, input_mask, segment_ids, label_ids)
    global graph
    with graph.as_default():
        result = {}
        result['code'] = 0
        try:
            sentence = request.args['query']
            result['query'] = sentence
            start = datetime.now()
            if (len(sentence) < 2):
                print(sentence)
                result['data'] = (['O'] * len(sentence))
                return json.dumps(result)
            sentence = tokenizer.tokenize(sentence)
            (input_ids, input_mask, segment_ids, label_ids) = convert(sentence)
            feed_dict = {input_ids_p: input_ids, input_mask_p: input_mask}
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            print(pred_label_result)
            result['data'] = pred_label_result
            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))
            return json.dumps(result)
        except:
            result['code'] = (- 1)
            result['data'] = 'error'
            return json.dumps(result)

def online_predict():
    '\n    do online prediction. each time make prediction for one instance.\n    you can change to a batch if you want.\n\n    :param line: a list. element is: [dummy_label,text_a,text_b]\n    :return:\n    '

    def convert(line):
        feature = convert_single_example(0, line, label_list, max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, max_seq_length))
        label_ids = np.reshape([feature.label_ids], (batch_size, max_seq_length))
        return (input_ids, input_mask, segment_ids, label_ids)
    global graph
    with graph.as_default():
        sentence = '北京天安门'
        start = datetime.now()
        if (len(sentence) < 2):
            print(sentence)
        sentence = tokenizer.tokenize(sentence)
        (input_ids, input_mask, segment_ids, label_ids) = convert(sentence)
        feed_dict = {input_ids_p: input_ids, input_mask_p: input_mask}
        pred_ids_result = sess.run([pred_ids], feed_dict)
        pred_label_result = convert_id_to_label(pred_ids_result, id2label)
        print(pred_label_result)
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

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    '\n    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中\n    :param ex_index: index\n    :param example: 一个样本\n    :param label_list: 标签列表\n    :param max_seq_length:\n    :param tokenizer:\n    :param mode:\n    :return:\n    '
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    if (not os.path.exists(os.path.join(model_dir, 'label2id.pkl'))):
        with open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
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
if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=12345)
