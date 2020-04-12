"""
main function for DL training.
"""
from __future__ import print_function

import datetime
import os
import sys
import time
import math
import threading
from multiprocessing import Process

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as config
from data_utils import input_fn_tfrecord

from WideDeep_fp16_huifeng import WideDeep

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import allreduce
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

rank_size = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID'))
#num_gpu = config.num_gpu
num_gpu = rank_size
mode = 'train'
algo='WideDeep'


data_para = {
    'batch_size': int(sys.argv[1]),#config.batch_size,
    'num_gpu': num_gpu
}
train_para = {
    'pos_weight': 1.0,
    'n_epoch': config.n_epoches,
    'train_per_epoch': config.train_size/5,
    'test_per_epoch': config.test_size,
    'batch_size': data_para['batch_size'],
    'early_stop_epochs': 40 
}


# set PIN model param
width = 1000
depth = 2
ls = [width] * depth
ls.append(1)
la = ['relu'] * depth
la.append(None)
lk = [.8] * depth
lk.append(1.)
model_param = {
    'init': 'xavier',
    'num_inputs': config.num_features,
    'input_dim': config.num_inputs,
    'real_inputs': config.num_features,
    'multi_hot_flags': config.multi_hot_flags,
    'multi_hot_len': config.multi_hot_len,
    'norm': False,
    'learning_rate': 5e-4,
    'embed_size': 80,
    'l2_v': 1e-4,
    'layer_sizes': ls,
    'layer_acts': la,
    'layer_keeps': lk,
    'layer_l2': None,
    'net_sizes': [80],
    'net_acts': ['relu', None],
    'net_keeps': [0.8],
    'wide': True,
    'layer_norm': True,
    'sub_layer_norm': False
}


def write_log(log_path, _line, echo=False):
    with open(log_path, 'a') as log_in:
        log_in.write(_line + '\n')
        if echo:
            print(_line)


def metric(log_path, batch_auc, y, p, name='ctr', cal_prob=None):
    y = np.array(y)
    p = np.array(p)

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
    # print('train auc: %g\tavg ctr: %g' % (batch_auc, y.mean()))

    ll = log_loss(y, p) * afterlen / orilen;
    q = y.mean()
    ne = ll / (-1 * q * np.log(q) - (1 - q) * np.log(1 - q))
    rig = 1 - ne

    if log_path:
        log = '%s\t%g\t%g\t%g\t%g' % (name, batch_auc, auc, ll, ne)
        write_log(log_path, log)
    print('avg %s on p: %g\teval auc: %g\tlog loss: %g\tne: %g\trig: %g' %
          (name, q, auc, ll, ne, rig))
    return auc


def evaluate_batch(sess, _model, num_gpu, batch_ids, batch_ws, batch_ys):
    if num_gpu == 1:
        model = _model
        feed_dict = {model.eval_id_hldr: batch_ids, model.eval_wt_hldr: batch_ws}
        _preds_ = sess.run(fetches=model.eval_preds, feed_dict=feed_dict)
        batch_preds = [_preds_.flatten()]
    else:
        model = _model
        feed_dict = {model.eval_id_hldr: batch_ids, model.eval_wt_hldr: batch_ws}
        _preds_ = sess.run(fetches=model.eval_preds, feed_dict=feed_dict)
        batch_preds = [_preds_.flatten()]
    return batch_preds

def get_optimizer(optimizer_array, global_step):
    opt = optimizer_array[0].lower()
    #if algo == 'DCN_T':
    lr = tf.train.exponential_decay(learning_rate=optimizer_array[1], global_step=global_step, decay_rate=optimizer_array[3], decay_steps=optimizer_array[4], staircase=True)
    #else:
    #lr = optimizer_array[1]
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        eps = optimizer_array[2]
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        init_val = optimizer_array[2]
        return tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=init_val)
    elif opt == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate=lr, initial_accumulator_value=optimizer_array[2],l1_regularization_strength=optimizer_array[3],l2_regularization_strength=optimizer_array[4])

def evaluate(sess, num_gpu,  model): # id_hldr, wt_hldr, eval_preds):
    preds = []
    labels = []
    line_cnt =0
    start_time = time.time()
    number_of_batches = ((train_para['test_per_epoch'] + train_para['batch_size'] - 1) /
                         train_para['batch_size'])
    print("%d batches in test set." % number_of_batches)
    #for _batch in range(1, number_of_batches + 1):
    epoch_finished = False
    test_dataset = input_fn_tfrecord(config.test_record, config.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()

    next_element = test_iterator.get_next()
    while not epoch_finished:
        # test_ids, test_wts, test_labels, epoch_finished = test_gen.next()
        test_batch_features = sess.run(next_element)
        test_ids = test_batch_features['feat_ids'].reshape((config.batch_size, config.num_inputs))
        test_wts = test_batch_features['feat_vals'].reshape((config.batch_size, config.num_inputs))
        test_labels = test_batch_features['label']

        line_cnt += test_labels.shape[0]
        preds.append(np.squeeze(evaluate_batch(sess, model,  num_gpu, test_ids, test_wts, test_labels)))
        labels.append(np.squeeze(test_labels))
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    print("evaluate time: %f sec" % (time.time() - start_time))
    return labels,  preds


def evaluate(sess, num_gpu,  model): # id_hldr, wt_hldr, eval_preds):
    preds = []
    labels = []
    line_cnt =0
    start_time = time.time()
    number_of_batches = ((train_para['test_per_epoch'] + train_para['batch_size'] - 1) /
                         train_para['batch_size'])
    print("%d batches in test set." % number_of_batches)
    #for _batch in range(1, number_of_batches + 1):
    epoch_finished = False
    test_dataset = input_fn_tfrecord(config.test_tag, config.batch_size / config.line_per_sample)
    test_iterator = test_dataset.make_one_shot_iterator()

    next_element = test_iterator.get_next()
    while not epoch_finished:
        # test_ids, test_wts, test_labels, epoch_finished = test_gen.next()
        try:
            test_batch_features = sess.run(next_element)
            test_ids = test_batch_features['feat_ids'].reshape((-1, config.num_inputs))
            test_wts = test_batch_features['feat_vals'].reshape((-1, config.num_inputs))
            test_labels = test_batch_features['label'].reshape((-1, ))

            line_cnt += test_labels.shape[0]
            preds.append(np.squeeze(evaluate_batch(sess, model,  num_gpu, test_ids, test_wts, test_labels)))
            labels.append(np.squeeze(test_labels))
        except tf.errors.OutOfRangeError:
            print("end of test trainset")
            epoch_finished = True

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    print("evaluate time: %f sec" % (time.time() - start_time))
    return labels,  preds


def build_model(graph, _input_d):
    seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
    input_dim = config.num_features
    num_inputs = config.num_inputs

    #model = DCN_T([input_dim, num_inputs, config.multi_hot_flags,
    #               config.multi_hot_len],
    #              _input_d,
    #              [80, 3, [1024, 512, 256, 128], 'relu'],
    #              ['uniform', -0.001, 0.001, seeds[4:14], None],
    #              ['adam', 1e-4, 5e-8, 0.6, 5],
    #              [0.7, 8e-5, 1e-8]
    #              )
    #model = WideDeep([input_dim, num_inputs, config.multi_hot_flags,
    #               config.multi_hot_len],
    #              _input_d,
    #              [80, 3, [1024, 512, 256, 128], 'relu'],
    #              ['uniform', -0.001, 0.001, seeds[4:14], None],
    #              ['adam', 1e-4, 5e-8, 0.6, 5],
    #              [0.7, 8e-5, 1e-8]
    #              )
    model =WideDeep(graph, [input_dim, num_inputs, config.multi_hot_flags,
                   config.multi_hot_len],
                  [80, [1024, 512, 256, 128], 'relu'],
                  ['uniform', -0.01, 0.01, seeds[4:14], None],
                  [['adam', 5e-4, 5e-8, 0.6, 5],['ftrl', 1e-2, 1, 1e-8, 1e-8]],
                  [1.0, 1e-7],
                  _input_d
                  )
    print('mode:%s, batch size: %d, buf size: %d, eval size: %d' % (
        mode, batch_size, buf_size, eval_size))

    write_log(log_file, model.log, True)
    return model

def average_gradients(gpu_grads):
    #if(len(gpu_grads) == 1):
    #    return gpu_grads[0]

    avg_grads = []
    for grad_and_vars in zip(*gpu_grads):
        grads = []

        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        all_grad = tf.concat(grads, 0)
        avg_grad = tf.reduce_mean(all_grad, 0, keep_dims=False)

        v = grad_and_vars[0][1]
        grad_and_var = (avg_grad, v)
        avg_grads.append(grad_and_var)

    return avg_grads

def build_graph(graph, input_data):
    # tf.reset_default_graph()
    with tf.device(0):
        with tf.variable_scope(tf.get_variable_scope()):
            global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                               initializer=tf.constant_initializer(0), trainable=False)
            model = build_model(graph, input_data)
    return model, model.train_op


def train_batch(sess, num_gpu, _model, train_op):
    if num_gpu >= 1:
        model = _model
        fetche = [train_op]
        #fetche = [i for i in train_op]
        if True:
           fetche = fetche + [model.deep_loss, model.log_loss, model.l2_loss, model.train_preds, model.lbl_hldr]
           _, _deeploss_, _log_loss_, _l2_loss_, _preds_, _train_labels= sess.run(fetches=fetche)
           #fetch = []
           #fetche = [fetche[1]] + [ model.wide_loss, model.log_loss, model.l2_loss, model.train_preds, model.lbl_hldr]
           #_,_wideloss_, _log_loss_, _l2_loss_, _preds_, _train_labels= sess.run(fetches=fetche)
           _loss_ = _deeploss_
    else:
        pass 
    return _loss_, _log_loss_, _l2_loss_, _preds_, _train_labels

def create_dirs(dir):
    """create dir recursively as needed."""
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    display_step = 1
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    input_dim = config.num_features
    num_inputs = config.num_inputs
    print("input_dim={}, num_inputs={}".format(input_dim, num_inputs))

    # timestamp = sys.argv[1]
    # tag = timestamp + '-' + algo

    tag = algo
    Base_path = config.BASE_DIR
    log_path = os.path.join(Base_path, 'log/')
    create_dirs(log_path)
    log_file = os.path.join(log_path, tag)
    pickle_model_path = os.path.join(Base_path,
                                     'model/pickle_model/')
    create_dirs(pickle_model_path)
    print("log file: ", log_file)

    batch_size = data_para['batch_size']
    buf_size = train_para['train_per_epoch']
    eval_size = data_para['batch_size']
    early_stop_epochs = train_para['early_stop_epochs']

    metric_best = 0
    metric_best_epoch = -1

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    
    # for npu
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["mix_compile_mode"].b = False  #开启混合计算，根据实际情况配置
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    custom_op.parameter_map["hcom_parallel"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")    
 
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    
    global_start_time = time.time()
    #with tf.device('/cpu:0'):
    graph = tf.Graph()
    with graph.as_default():
    #with tf.device(0):
        #train_dataset = input_fn_tfrecord(config.train_record, batch_size=config.batch_size * num_gpu / config.line_per_sample,
        #                                  perform_shuffle=True, num_epochs=config.n_epoches)

        #tf.random.set_random_seed(1234)
        train_dataset = input_fn_tfrecord(config.train_tag, batch_size=int(sys.argv[1]) / config.line_per_sample,
                                          perform_shuffle=True, num_epochs=config.n_epoches)
        print('batch size:', int(sys.argv[1]) * num_gpu / config.line_per_sample)
        #rank_size = int(os.getenv('RANK_SIZE'))
        if mode == 'train' and rank_size > 1:
            rank_size = int(os.getenv('RANK_SIZE'))
            rank_id = int(os.getenv('DEVICE_INDEX'))
            print("ranksize = %d, rankid = %d" % (rank_size, rank_id))
            train_dataset = train_dataset.shard(rank_size, rank_id)

        iterator = train_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        input_data = [tf.reshape(next_element['label'], [-1]),
                      tf.reshape(next_element['feat_ids'], [-1, 39]),
                      tf.reshape(next_element['feat_vals'], [-1, 39])]

    if num_gpu >= 1:
        model, opt = build_graph(graph, input_data)
    #else:
    #    exit(0)

    with tf.Session(graph=graph, config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.writer_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        #saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint(Base_path + 'model', 'WideDeep-checkpoint'))

        print('model initialized')

        if mode == 'train':

            start_time = time.time()
            est_epoch_batches = int((train_para['train_per_epoch'] +
                                     train_para['batch_size']*num_gpu - 1) / (train_para['batch_size']*num_gpu))
            est_tot_batches = train_para['n_epoch'] * est_epoch_batches
            _epoch = 1
            train_finished = False
            while _epoch < train_para['n_epoch'] + 1 and not train_finished:
                _epoch_start_time = time.time()
                epoch_loss = []
                epoch_labels = []
                epoch_preds = []
                epoch_auc = -1
                epoch_finished = False
                epoch_sample_num = 0
                epoch_finished_batches = 0
                cnt = 0
                saver = tf.train.Saver()
                #saver.save(sess, Base_path + 'model/%s' % tag,
                #                        global_step=cnt, latest_filename='%s-checkpoint' % tag)
                while not epoch_finished and not train_finished:
                    try:
                        _loss, _log_loss, _l2_loss, p, _labels = train_batch(sess, num_gpu, model, opt)
                        if (config.one_step):
                            if (cnt < config.one_step):
                                cnt = cnt+1
                                print('step %d: loss = %f : log_loss = %f : l2_loss = %f : device_id = %d' % (cnt, np.array(_loss).mean(), np.array(_log_loss).mean(), np.array(_l2_loss).mean(), rank_id))
                                #saver = tf.train.Saver()
                                #saver.save(sess, Base_path + 'model/%s' % tag,
                                #        global_step=cnt, latest_filename='%s-checkpoint' % tag)
                                #epoch_finished = True
                                #train_finished = True
                                #break 
                            else:
                                epoch_finished = True
                                train_finished = True
                                #saver = tf.train.Saver()
                                #saver.save(sess, Base_path + 'model/%s' % tag,
                                #        global_step=cnt, latest_filename='%s-checkpoint' % tag)
                                print('step %d: loss = %f : log_loss = %f : l2_loss = %f : device_id = %d' % (config.one_step, np.array(_loss).mean(), np.array(_log_loss).mean(), np.array(_l2_loss).mean(),rank_id))
                                break
                        epoch_loss.append(_loss)
                        epoch_labels.append(_labels)
                        epoch_preds.extend(p)
                        epoch_finished_batches += 1
                        epoch_sample_num += _labels.shape[0]
                        if epoch_finished_batches % (display_step / num_gpu) == 0: # print step
                            if _epoch:
                                avg_loss = np.array(epoch_loss).mean()
                                epoch_auc = 0#roc_auc_score(epoch_labels, epoch_preds)
                                elapsed = int(time.time() - start_time)
                                finished_batches = (_epoch-1) * est_epoch_batches + epoch_finished_batches
                                eta = int(1.0 * (est_tot_batches - finished_batches) /
                                          finished_batches * elapsed)
                                epoch_labels = []
                                epoch_preds = []
                                epoch_loss = []
                                print('epoch %3d/%3d - batch %5d: loss = %f, auc = %f, device_id = %d | elapsed : %s, ETA : %s' % (
                                    _epoch, train_para['n_epoch'], epoch_finished_batches, avg_loss, epoch_auc, rank_id,
                                    str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta))))
                                avg_loss = 0
                            else:
                                elapsed = int(time.time() - start_time)
                                finished_batches = (_epoch-1) * est_epoch_batches + epoch_finished_batches
                                eta = int(1.0 * (est_tot_batches - finished_batches) /
                                          finished_batches * elapsed)
                                print('epoch %3d/%3d - batch %5d: | elapsed : %s, ETA : %s' % (
                                    _epoch, train_para['n_epoch'], epoch_finished_batches,
                                    str(datetime.timedelta(seconds=elapsed)), str(datetime.timedelta(seconds=eta))))
                        if epoch_finished_batches % est_epoch_batches == 0 or epoch_finished:
                            epoch_finished = True
                            print('epoch %d train time = %.3f sec, #train sample = %d' %
                                  (_epoch, time.time() - _epoch_start_time, epoch_sample_num))

                            # ======== comment this block if no testset available ========
                            print("== starting evaluate == ")
                            eval_labels, eval_preds = evaluate(sess, num_gpu, model)
                            eval_auc = metric(log_file, epoch_auc, eval_labels, eval_preds)
                            print("== finished evaluate == ")
                            # ============================================================
                            print('epoch %d total time = %s' %
                                  (_epoch, str(time.time() - _epoch_start_time)))

                            if eval_auc >= metric_best:
                                metric_best = eval_auc
                                metric_best_epoch = _epoch
                                saver = tf.train.Saver()
                                saver.save(sess, Base_path + 'model/%s' % tag,
                                           global_step=_epoch,
                                           latest_filename='%s-checkpoint' % tag)
                                print("current best auc: ", metric_best, " best_epoch: ", metric_best_epoch)
                            else:
                                if _epoch - metric_best_epoch >= early_stop_epochs:
                                    print("the model will be early stopped: current epoch:", _epoch)
                                    log_data = "best epoch: %d\t best performance:%g\n" % (metric_best_epoch, metric_best)
                                    log_data += "model_saved to %s\n" % (Base_path + 'model/%s' % tag)
                                    write_log(log_file, log_data, echo=True)

                                    print("save complete for epoch %d" % _epoch)
                                    train_finished = True
                                    break
                            _epoch += 1
                    except tf.errors.OutOfRangeError as e:
                        print("end of training dataset")
                        print("epoch %3d finished ..." % _epoch)
                        train_finished = True
            writer.close()

            # print('converting to checkpoint model pb format..')
            # freeze_graph(model_folder=Base_path + '/model/',
            #              latest_filename='%s-checkpoint' % tag, tag=tag)
            # freeze_graph(model_folder=Base_path + '/model/',
            #              latest_filename='%s-checkpoint' % tag, tag=None)
            # print('finish converting model to pb format')

        elif mode == 'test':
            pass
            # labels, preds = evaluate(sess, num_gpu, model)
            # metric(None, -1, labels, preds)
        tf.train.write_graph(sess.graph, config.graph_path, 'widedeep16_graph.pbtxt', as_text=True)
        sess.close()
