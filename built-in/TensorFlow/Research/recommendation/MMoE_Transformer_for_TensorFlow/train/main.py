"""
main function for DL training.
"""
from __future__ import print_function

import argparse
import datetime
import math
import os
import sys
import time
import random
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib

from models.mmoe_transformer import TransformerMMOE


def write_log(log_path, _line, echo=False):
    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print(_line)


def metric(log_path, y, p, name, filter=None, cal_prob=None, return_type='auc', pre_name='ctr'):
    """
    :param log_path: file path to write log info.
    :param y: list of label.
    :param p: list of predicted ctr.
    :param name: 'train' or 'test'. (prefix of log info)
    :param cal_prob: float, (0, 1]
    :param return_type: 'auc' or 'log_loss'
    :return:
    """
    from sklearn.metrics import roc_auc_score, log_loss
    y = np.array(y)
    p = np.array(p)

    if filter is not None:
        filter_mask = filter == 1
        y = y[filter_mask]
        p = p[filter_mask]

    if cal_prob:
        if cal_prob <= 0 or cal_prob > 1:
            raise ValueError('please ensure cal_prob is in (0,1]!')
        p /= (p + (1 - p) / cal_prob)
    auc = roc_auc_score(y, p)
    orilen = len(p)

    ind = np.where((p > 0) & (p < 1))[0]
    # print(len(ind))
    y = y[ind]
    p = p[ind]
    afterlen = len(p)
    ll = log_loss(y, p) * afterlen / orilen
    q = y.mean()
    # ne = ll / (-1 * q * np.log(q) - (1 - q) * np.log(1 - q))
    # rig = 1 - ne

    if log_path:
        # log = '[%s] %.6g\t%.6g\t%.6g' % (name, auc, ll, ne)
        log = '[%s] positive ratio on [%s]: %8.6f|auc: %8.6f|log loss: %8.6f' % (name, pre_name, q, auc, ll)
        write_log(log_path, log, echo=True)

    if return_type.lower() == 'auc':
        return auc
    return ll


def evaluate(sess, model):
    preds_1 = []
    preds_2 = []
    _labels_1 = []
    _labels_2 = []

    _start_time = time.time()
    line_cnt = 0
    test_finished = False
    while not test_finished:
        try:
            eval_preds_1, eval_preds_2, test_labels_1, test_labels_2=\
                sess.run([model.eval_preds_ctr, model.eval_preds_cvr,\
                          model.eval_ctr_label, model.eval_cvr_label])
            preds_1.extend(eval_preds_1)
            preds_2.extend(eval_preds_2)

            _labels_1.extend(test_labels_1)
            _labels_2.extend(test_labels_2)
        
        except tf.errors.OutOfRangeError as e:
            print("end of evaluating dataset")
            test_finished = True
    
    print("evaluate time: %f sec" % (time.time() - _start_time))
    return np.array(_labels_1), np.array(_labels_2), np.array(preds_1), np.array(preds_2)


def create_dirs(dir):
    """create dir recursively as needed."""
    if not os.path.exists(dir):
        os.makedirs(dir)

def input_fn_tfrecord(data_path,
                      file_pattern, 
                      batch_size=16000,
                      num_epochs=1,
                      num_parallel=16,
                      perform_shuffle=True):

    def extract_fn(data_record):
        features = {}
        for key, _shape in int_shape_d.items():
            num = 1
            for n in _shape:
                num = num * n
            print("key: {}, num: {} ".format(key, num))
            features[key] = tf.FixedLenFeature(shape=(num, ), dtype=tf.int64)
        for key,_shape in float_shape_d.items():
            num = 1
            for n in _shape:
                num = num * n
            print("key: {}, num: {} ".format(key, num))
            features[ key ] = tf.FixedLenFeature(shape=(num, ), dtype=tf.float32)
        sample = tf.parse_single_example(data_record, features)
        return sample

    def reshape_fn(sample):
        for key, _shape in list(int_shape_d.items()) + list(float_shape_d.items()):
            if len(_shape) > 1:
                sample[key] = tf.reshape(sample[key], [-1] + _shape[1:] )
            else:
                sample[key] = tf.reshape(sample[key], [-1] )
        for key,_shape in int_shape_d.items():
            sample[key] = tf.cast(sample[key], tf.int32)
        return sample


    data_args_path = os.path.join(data_path, "data_args.json")
    with open(data_args_path, "r", encoding="utf-8") as file_in:
        for line in file_in:
            data_args_dict = json.loads(line.strip("\n"))
    # 
    line_example = data_args_dict["line_example"]
    int_shape_d = data_args_dict["int_shape"]
    float_shape_d = data_args_dict["float_shape"]
    # 
    line_num = int(batch_size/line_example)
    # 
    all_files = os.listdir(data_path)
    files = [os.path.join(data_path, f) for f in all_files if f.startswith(file_pattern)]
    dataset = tf.data.TFRecordDataset(files).map(extract_fn, num_parallel_calls=num_parallel).batch(line_num)
    dataset = dataset.map(reshape_fn, num_parallel_calls=num_parallel)
    # 
    dataset = dataset.repeat(num_epochs)
    if perform_shuffle:
        dataset = dataset.shuffle(batch_size)
    # 
    return dataset


if __name__ == '__main__':
    tf.set_random_seed(2017)
    np.random.seed(2017)
    random.seed(2017)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="tag to identify model", default="tag")
    parser.add_argument("--config_name", help="config file name", default="hiad_config")
    parser.add_argument("--train_with_evaluate", action="store_true", help="train with evaluate")
    parser.add_argument("--npu_mode", action="store_true", help="npu mode")
    parser.add_argument("--iterations_per_loop_mode", action="store_true", help="iterations per loop mode")
    args = parser.parse_args()

    npu_mode = args.npu_mode
    iterations_per_loop_mode = args.iterations_per_loop_mode
    train_with_evaluate = args.train_with_evaluate
    if npu_mode:
        from npu_bridge.npu_init import *
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

    config = importlib.import_module('config.' + args.config_name)

    batch_size = config.batch_size
    deep = config.deep
    cross = config.cross
    embed_size = config.embed_size
    l2 = config.l2
    l1 = config.l1
    lr = config.lr
    decay_step = config.decay_step
    decay_rate = config.decay_rate
    num_experts = config.num_experts
    num_expert_units = config.num_expert_units
    keep_prob = config.keep_prob

    base_path = config.base_dir
    num_parallel = config.num_workers
    data_para = config.data_para
    train_para = config.train_para
    transformer_params = config.transformer_params
    
    iter_per_epoch = math.ceil(data_para['train_size'] / batch_size)

    train_data_path = data_para["train_path"]
    train_dataset = input_fn_tfrecord(train_data_path,
                                      file_pattern="train", 
                                      batch_size=batch_size,
                                      num_epochs=train_para['n_epoch'], 
                                      num_parallel=num_parallel)
    train_iterator = train_dataset.make_initializable_iterator()

    test_data_path = data_para["test_path"]
    test_dataset = input_fn_tfrecord(test_data_path,
                                      file_pattern="test", 
                                      batch_size=batch_size,
                                      num_epochs=1, 
                                      num_parallel=num_parallel)
    test_iterator = test_dataset.make_initializable_iterator()
    
    # max_length
    print("input_dim={}, fields_num={}".format(data_para['input_dim'], data_para['fields_num']))

    algo = 'SBDNN-shuffle-{}'.format(config.shuffle_seed_idx)
    tag = algo + '-' + str(batch_size) + '-' + str(embed_size) + '-' + str(lr) + '-' + args.tag

    log_path = os.path.join(base_path, 'log/')
    create_dirs(log_path)
    log_file = os.path.join(log_path, tag)
    print("log file: ", log_file)
    model_path = os.path.join(base_path, 'model/')
    create_dirs(model_path)
    if config.shuffle_seed_idx:
        current_time = time.time()
        seeds = [int(current_time / i) for i in range(1, 101)]
        random.shuffle(seeds)
    else:
        seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
                 0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
    
    early_stop_steps = train_para['early_stop_steps']
    metric_best = 0

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    if npu_mode:
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
        custom_op.parameter_map["dynamic_input"].b = True
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile") 
        if iterations_per_loop_mode:
            custom_op.parameter_map["iterations_per_loop"].i = 10
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    global_start_time = time.time()

    model = TransformerMMOE(train_iterator,
                            test_iterator,
                            transformer_params,
                            iter_per_epoch,
                            [data_para['input_dim'], data_para["fields_num"]],
                            embed_size,
                            [num_expert_units, num_experts, 'relu'],
                            [deep, cross, 'relu'],
                            ['uniform', -0.001, 0.001, seeds[4:14], None],
                            ['adam', lr, 5e-8, decay_rate, decay_step],
                            [keep_prob, l2, l1],
                            max_seq_len=300,
                            batch_norm=True,
                            npu_mode=npu_mode
                            )
    write_log(log_file, model.log, True)

    print('model initialized')

    with tf.Session(config=sess_config) as sess:
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        start_time = time.time()
        each_start_time = start_time
        
        train_finished = False 
        current_steps = 1
        while not train_finished:
            try:
                if iterations_per_loop_mode:
                    train_op = util.set_iteration_per_loop(sess, model.optmzr, 10)
                    _, loss = sess.run([train_op, model.loss])
                else:
                    _, loss = sess.run([model.optmzr, model.loss])

                if current_steps % 100 == 0:
                    
                    time_per_step = (time.time() - each_start_time) / 100
                    print('iter %5d - loss : %.3f - per step time : %.3f sec' % (
                        current_steps, loss, time_per_step))
                    each_start_time = time.time()

                    if train_with_evaluate:
                        sess.run(model.test_dataset_iterator.initializer)
                        # save checkpoint if metrics get better
                        print("== start evaluating test set == ")
                        eval_labels_1, eval_labels_2, eval_preds_ctr, eval_preds_cdr = evaluate(sess, model)
                        eval_auc_ctr = metric(log_file, eval_labels_1, eval_preds_ctr, 'test', pre_name="ctr")
                        eval_auc_cvr = metric(log_file, eval_labels_2, eval_preds_cdr, 'test', pre_name="cdr")
                        print("== finished evaluate == ")

                        # TODO here use auc of ctcvr to early stop
                        if eval_auc_ctr >= metric_best:
                            metric_best = eval_auc_ctr
                            metric_best_step = current_steps
                            saver.save(sess, os.path.join(model_path, tag),
                                    global_step=current_steps, latest_filename='%s-checkpoint' % tag)
                            print("current best auc: ", metric_best,
                                " best_step: ", metric_best_step)
                        else:
                            if current_steps - metric_best_step >= early_stop_steps:
                                print("the model will be early stopped: current step:", current_steps)
                                log_data = "best step: %d\t best performance:%g\n" % (metric_best_step, metric_best)
                                log_data += "model_saved to %s\n" % (os.path.join(model_path, tag))
                                write_log(log_file, log_data, echo=True)
                                print("save complete for step %d" % current_steps)
                                break

                current_steps += 1

            except tf.errors.OutOfRangeError as e:
                print("end of training dataset")
                train_finished = True

        print('train time = %.3f sec, #num epoch = %d, loss = %.3f' %
                (time.time() - start_time, train_para['n_epoch'], loss))

