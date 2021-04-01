
'\nCopyright 2018 The Google AI Language Team Authors.\nBASED ON Google_BERT.\nreference from :zhoukaiyin/\n\n@Author:Macan\n'
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import collections
import os
import numpy as np
import tensorflow as tf
import codecs
import pickle
from bert_base.train import tf_metrics
from bert_base.bert import modeling
from bert_base.bert import optimization
from bert_base.bert import tokenization
from bert_base.train.models import create_model, InputFeatures, InputExample
from bert_base.server.helper import set_logger
__version__ = '0.1.0'
__all__ = ['__version__', 'DataProcessor', 'NerProcessor', 'write_tokens', 'convert_single_example', 'filed_based_convert_examples_to_features', 'file_based_input_fn_builder', 'model_fn_builder', 'train']
logger = set_logger('NER Training')

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

    @classmethod
    def _read_data(cls, input_file):
        'Reads a BIO data.'
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if (len(tokens) == 2):
                    words.append(tokens[0])
                    labels.append(tokens[1])
                elif (len(contends) == 0):
                    l = ' '.join([label for label in labels if (len(label) > 0)])
                    w = ' '.join([word for word in words if (len(word) > 0)])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                if contends.startswith('-DOCSTART-'):
                    words.append('')
                    continue
            return lines

class NerProcessor(DataProcessor):

    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, 'train.txt')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, 'dev.txt')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, 'test.txt')), 'test')

    def get_labels(self, labels=None):
        if (labels is not None):
            try:
                if (os.path.exists(labels) and os.path.isfile(labels)):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    self.labels = labels.split(',')
                self.labels = set(self.labels)
            except Exception as e:
                print(e)
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        elif (len(self.labels) > 0):
            self.labels = self.labels.union(set(['X', '[CLS]', '[SEP]']))
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                pickle.dump(self.labels, rf)
        else:
            self.labels = ['O', 'B-TIM', 'I-TIM', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'X', '[CLS]', '[SEP]']
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = ('%s-%s' % (set_type, i))
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        'Reads a BIO data.'
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if (len(tokens) == 2):
                    words.append(tokens[0])
                    labels.append(tokens[(- 1)])
                elif ((len(contends) == 0) and (len(words) > 0)):
                    label = []
                    word = []
                    for (l, w) in zip(labels, words):
                        if ((len(l) > 0) and (len(w) > 0)):
                            label.append(l)
                            self.labels.add(l)
                            word.append(w)
                    lines.append([' '.join(label), ' '.join(word)])
                    words = []
                    labels = []
                    continue
                if contends.startswith('-DOCSTART-'):
                    continue
            return lines

def write_tokens(tokens, output_dir, mode):
    '\n    将序列解析结果写入到文件中\n    只在mode=test的时候启用\n    :param tokens:\n    :param mode:\n    :return:\n    '
    if (mode == 'test'):
        path = os.path.join(output_dir, (('token_' + mode) + '.txt'))
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if (token != '**NULL**'):
                wf.write((token + '\n'))
        wf.close()

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
    '\n    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中\n    :param ex_index: index\n    :param example: 一个样本\n    :param label_list: 标签列表\n    :param max_seq_length:\n    :param tokenizer:\n    :param output_dir\n    :param mode:\n    :return:\n    '
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    if (not os.path.exists(os.path.join(output_dir, 'label2id.pkl'))):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for (i, word) in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if (m == 0):
                labels.append(label_1)
            else:
                labels.append('X')
    if (len(tokens) >= (max_seq_length - 1)):
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append('[CLS]')
    segment_ids.append(0)
    label_ids.append(label_map['[CLS]'])
    for (i, token) in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
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
    if (ex_index < 5):
        logger.info('*** Example ***')
        logger.info(('guid: %s' % example.guid))
        logger.info(('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens])))
        logger.info(('input_ids: %s' % ' '.join([str(x) for x in input_ids])))
        logger.info(('input_mask: %s' % ' '.join([str(x) for x in input_mask])))
        logger.info(('segment_ids: %s' % ' '.join([str(x) for x in segment_ids])))
        logger.info(('label_ids: %s' % ' '.join([str(x) for x in label_ids])))
    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
    write_tokens(ntokens, output_dir, mode)
    return feature

def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
    '\n    将数据转化为TF_Record 结构，作为模型数据输入\n    :param examples:  样本\n    :param label_list:标签list\n    :param max_seq_length: 预先设定的最大序列长度\n    :param tokenizer: tokenizer 对象\n    :param output_file: tf.record 输出路径\n    :param mode:\n    :return:\n    '
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ((ex_index % 5000) == 0):
            logger.info(('Writing example %d of %d' % (ex_index, len(examples))))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(feature.input_ids)
        features['input_mask'] = create_int_feature(feature.input_mask)
        features['segment_ids'] = create_int_feature(feature.segment_ids)
        features['label_ids'] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {'input_ids': tf.FixedLenFeature([seq_length], tf.int64), 'input_mask': tf.FixedLenFeature([seq_length], tf.int64), 'segment_ids': tf.FixedLenFeature([seq_length], tf.int64), 'label_ids': tf.FixedLenFeature([seq_length], tf.int64)}

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if (t.dtype == tf.int64):
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params['batch_size']
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch((lambda record: _decode_record(record, name_to_features)), batch_size=batch_size, num_parallel_calls=8, drop_remainder=True))
        d = d.prefetch(buffer_size=4)
        return d
    return input_fn

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, args):
    '\n    构建模型\n    :param bert_config:\n    :param num_labels:\n    :param init_checkpoint:\n    :param learning_rate:\n    :param num_train_steps:\n    :param num_warmup_steps:\n    :param use_tpu:\n    :param use_one_hot_embeddings:\n    :return:\n    '

    def model_fn(features, labels, mode, params):
        if mode != tf.estimator.ModeKeys.TRAIN:
            exit(0)
        logger.info('*** Features ***')
        for name in sorted(features.keys()):
            logger.info(('  name = %s, shape = %s' % (name, features[name].shape)))
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        label_ids = features['label_ids']
        print('shape of input_ids', input_ids.shape)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, logits, trans, pred_ids) = create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)
        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        output_spec = None
        if (mode == tf.estimator.ModeKeys.TRAIN):
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=args.save_summary_steps)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        
        return output_spec
    return model_fn

def get_last_checkpoint(model_path):
    if (not os.path.exists(os.path.join(model_path, 'checkpoint'))):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if (len(line) != 2):
                continue
            if (line[0] == 'model_checkpoint_path'):
                last = line[1][2:(- 1)]
                break
    return last

def adam_filter(model_path):
    '\n    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的\n    :param model_path: \n    :return: \n    '
    last_name = get_last_checkpoint(model_path)
    if (last_name is None):
        return
    sess = tf.Session(config=npu_session_config_init())
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, (last_name + '.meta')))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if (('adam_v' not in var.name) and ('adam_m' not in var.name)):
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    processors = {'ner': NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    if (args.max_seq_length > bert_config.max_position_embeddings):
        raise ValueError(('Cannot use sequence length %d because the BERT model was only trained up to sequence length %d' % (args.max_seq_length, bert_config.max_position_embeddings)))
    if (args.clean and args.do_train):
        if os.path.exists(args.output_dir):

            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit((- 1))
    if (not os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)
    processor = processors[args.ner](args.output_dir)
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    session_config = npu_config_proto(config_proto=tf.ConfigProto(log_device_placement=False, inter_op_parallelism_threads=0, intra_op_parallelism_threads=0, allow_soft_placement=True))
    run_config = tf.estimator.RunConfig(model_dir=args.output_dir, save_summary_steps=500, save_checkpoints_steps=500, session_config=session_config)
    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if (args.do_train and args.do_eval):
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int((((len(train_examples) * 1.0) / args.batch_size) * args.num_train_epochs))
        if (num_train_steps < 1):
            raise AttributeError('training data is so small...')
        num_warmup_steps = int((num_train_steps * args.warmup_proportion))
        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', len(train_examples))
        logger.info('  Batch size = %d', args.batch_size)
        logger.info('  Num steps = %d', num_train_steps)
        eval_examples = processor.get_dev_examples(args.data_dir)
        logger.info('***** Running evaluation *****')
        logger.info('  Num examples = %d', len(eval_examples))
        logger.info('  Batch size = %d', args.batch_size)
    label_list = processor.get_labels()
    model_fn = model_fn_builder(bert_config=bert_config, num_labels=(len(label_list) + 1), init_checkpoint=args.init_checkpoint, learning_rate=args.learning_rate, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, args=args)
    params = {'batch_size': args.batch_size}
    estimator = tf.estimator.Estimator(model_fn, params=params, config=npu_run_config_init(run_config=run_config))
    if (args.do_train and args.do_eval):
        train_file = os.path.join(args.output_dir, 'train.tf_record')
        if (not os.path.exists(train_file)):
            filed_based_convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)
        train_input_fn = file_based_input_fn_builder(input_file=train_file, seq_length=args.max_seq_length, is_training=True, drop_remainder=True)
        eval_file = os.path.join(args.output_dir, 'eval.tf_record')
        if (not os.path.exists(eval_file)):
            filed_based_convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)
        eval_input_fn = file_based_input_fn_builder(input_file=eval_file, seq_length=args.max_seq_length, is_training=False, drop_remainder=False)
                
        import time
        class MyHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = 0
                self._loss_tensor = tf.get_default_graph().get_tensor_by_name('crf_loss/Mean:0')

            def after_create_session(self, session, coord):
                pass

            def before_run(self, run_context):
                self._begin = time.time()
                self._step += 1
                return tf.train.SessionRunArgs({'loss': self._loss_tensor})

            def after_run(self, run_context, run_values):
                self._end = time.time()
                cost = self._end - self._begin
                loss = run_values.results['loss']
                with open('output/run.log', 'a') as f:
                    print('step: %d, cost time: %.3fs, losses: %.3f' % (self._step, cost, loss))

            def end(self, session):
                pass

        myHook = MyHook()

        stepHook = tf.train.StopAtStepHook(5)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=[myHook, stepHook])      
        
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if args.do_predict:
        token_path = os.path.join(args.output_dir, 'token_test.txt')
        if os.path.exists(token_path):
            os.remove(token_path)
        with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for (key, value) in label2id.items()}
        predict_examples = processor.get_test_examples(args.data_dir)
        predict_file = os.path.join(args.output_dir, 'predict.tf_record')
        filed_based_convert_examples_to_features(predict_examples, label_list, args.max_seq_length, tokenizer, predict_file, args.output_dir, mode='test')
        logger.info('***** Running prediction*****')
        logger.info('  Num examples = %d', len(predict_examples))
        logger.info('  Batch size = %d', args.batch_size)
        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(input_file=predict_file, seq_length=args.max_seq_length, is_training=False, drop_remainder=predict_drop_remainder)
        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(args.output_dir, 'label_test.txt')

        def result_to_pair(writer):
            for (predict_line, prediction) in zip(predict_examples, result):
                idx = 0
                line = ''
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')
                len_seq = len(label_token)
                if (len(line_token) != len(label_token)):
                    logger.info(predict_line.text)
                    logger.info(predict_line.label)
                    break
                for id in prediction:
                    if (idx >= len_seq):
                        break
                    if (id == 0):
                        continue
                    curr_labels = id2label[id]
                    if (curr_labels in ['[CLS]', '[SEP]']):
                        continue
                    try:
                        line += (((((line_token[idx] + ' ') + label_token[idx]) + ' ') + curr_labels) + '\n')
                    except Exception as e:
                        logger.info(e)
                        logger.info(predict_line.text)
                        logger.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write((line + '\n'))
        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)
        from bert_base.train import conlleval
        eval_result = conlleval.return_report(output_predict_file)
        print(''.join(eval_result))
        with codecs.open(os.path.join(args.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))
    if args.filter_adam_var:
        adam_filter(args.output_dir)
