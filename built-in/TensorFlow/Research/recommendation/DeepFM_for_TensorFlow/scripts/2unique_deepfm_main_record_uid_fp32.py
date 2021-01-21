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
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
#import horovod.tensorflow as hvd
from npu_bridge.hccl import hccl_ops
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id


from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import ops
 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as config
from data_utils import input_fn_tfrecord


#from models.WideDeep import WideDeep
#from FMNN_huifeng import FMNN_v2
from FMNN_huifeng_multi import FMNN_v2
#imigrate sess
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager

mode = 'train'
algo='DeepFM'

def huifeng_debug():
    for i in range(10):
        print("huifeng_debug")


def write_log(log_path, _line, echo=False):
    if log_path is not None:
        with open(log_path, 'a') as log_in:
            log_in.write(_line + '\n')
            if echo:
                print(_line)


def metric(log_path, y, p, name='ctr', cal_prob=None):
    y = np.array(y)
    p = np.array(p)

    if cal_prob:
        if cal_prob <= 0 or cal_prob > 1:
            raise ValueError('please ensure cal_prob is in (0,1]!')
        p /= (p + (1 - p) / cal_prob)
    auc = roc_auc_score(y, p)
    orilen = len(p)

    ind = np.where((p > 0) & (p < 1))[0]
    print(len(ind),len(p),len(y))
    print(p)
    y = y[ind]
    p = p[ind]
    afterlen = len(p)
    # print('train auc: %g\tavg ctr: %g' % (batch_auc, y.mean()))

    ll = log_loss(y, p) * afterlen / orilen;
    q = y.mean()
    ne = ll / (-1 * q * np.log(q) - (1 - q) * np.log(1 - q))
    rig = 1 - ne

    if log_path:
        log = '%s\t%g\t%g\t%g' % (name, auc, ll, ne)
        write_log(log_path, log)
    print('avg %s on p: %g\teval auc: %g\tlog loss: %g\tne: %g\trig: %g' %
          (name, q, auc, ll, ne, rig))
    return auc


def get_optimizer(optimizer_array, global_step):
    opt = optimizer_array[0].lower()
    #if algo == 'DCN_T':
    #   lr = tf.train.exponential_decay(learning_rate=optimizer_array[1], global_step=global_step, decay_rate=optimizer_array[3], decay_steps=optimizer_array[4], staircase=True)
    #else:
    lr = optimizer_array[1]
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
    elif opt == 'lazyadam':
        lr = optimizer_array[1]
        eps = optimizer_array[2]
        return tf.contrib.opt.LazyAdamOptimizer(learning_rate=lr, epsilon=eps)


def build_model(_input_d):
    seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
             0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]
    num_comm_feat = config.common_feat_num
    dim_comm = config.common_dim

 
    model = FMNN_v2([num_comm_feat, dim_comm, config.multi_hot_flags,
                   config.multi_hot_len, int(config.general_feat_num/get_rank_size())+ 1, config.general_dim],
                  [80, [1024, 512 ,256 ,256], 'relu'],
                  ['uniform', -0.001, 0.001, seeds[4:14], None],
                  ['adam', 5e-4, 5e-8, 0.6, 5],
                  [0.7, 5e-6],
                  _input_d
                  )
    print('mode:%s, batch size: %d, eval size: %d' % (
        mode, batch_size, eval_size))

    write_log(log_file, model.log, True) if get_rank_id() == 0 else None
    return model

def allreduce(grads, average=True, compression=None, sparse_as_dense=True):
    if get_rank_size() == 1:
        print('aa', get_rank_size())
        return grads
    averaged_gradients = []
    with tf.name_scope("Allreduce"):
        for grad, var in grads:
            if grad is not None:
                if sparse_as_dense:
                    grad = tf.convert_to_tensor(grad)
                print('grad', grad, var)
                print('enter in','*'*1000)
                if compression is not None:
                    #avg_grad = grad
                    avg_grad = hccl_ops.allreduce(grad, reduction='sum')# average = average, compression = compression)
                    avg_grad /=get_rank_size()
                else:
                    #avg_grad = grad
                    avg_grad = hccl_ops.allreduce(grad,  reduction='sum')#average = average)
                    avg_grad /=get_rank_size()
                averaged_gradients.append((avg_grad, var))
            else:
                averaged_gradients.append((None, var))
    return averaged_gradients

def build_graph_ingraph(optimizer_array, input_data):
    # tf.reset_default_graph()
    with tf.variable_scope(tf.get_variable_scope()):
        global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                           initializer=tf.constant_initializer(0), trainable=False)
        model = build_model(input_data)
        opt = [get_optimizer(i, global_step) for i in optimizer_array]
        # tf.get_variable_scope().reuse_variables()
        # grads = opt.compute_gradients(model.loss)
        train_op = []
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #grads = opt[0].compute_gradients(model.loss)
            var_list = need_init_varlist(model) + [model.embedding_uniq_, model.W_uniq_]
            _grads = tf.gradients(model.loss,var_list)
            grads = []
            for i in range(len(var_list)-2):
                if _grads[i] != None:
                    grads.append((_grads[i], var_list[i]))
            print('grads:-------', grads, len(grads)) 
            dense_grads =  [grad for grad in grads if grad[1].name != "V_unique:0" and grad[1].name != "w_unique:0" and 
                           grad[1].name != "local_V:0" and grad[1].name != "local_W:0"]# and not isinstance(grad[0], tf.IndexedSlices)]

            dense_grads = allreduce(dense_grads)

            #''' 
            # back unique 1
            with ops.get_default_graph()._kernel_label_map({"Unique": "parallel"}):
                id_uniq_u, id_uniq_inver_all_ = tf.unique(tf.reshape((model.id_hldr_uniq_all-config.common_feat_num), [-1]), out_idx=tf.dtypes.int32)
            id_uniq_inver_all = tf.reshape(id_uniq_inver_all_, [-1, config.general_dim])
            # back unique 2 
            with ops.get_default_graph()._kernel_label_map({"Unique": "parallel"}):
                id_uniq_u2, id_uniq_inver_all_2 = tf.unique(tf.reshape((model.id_hldr_uniq_all-config.common_feat_num), [-1]), out_idx=tf.dtypes.int32)
            id_uniq_inver_all2 = tf.reshape(id_uniq_inver_all_2, [-1, config.general_dim])
            id_uniq_inver_ori = tf.gather(id_uniq_inver_all2, indices=model.batch_indices)
            id_uniq_inver = tf.reshape(id_uniq_inver_ori, [-1, model.dim_unique])
            id_uniq_local_mask = tf.where(tf.equal(tf.cast(get_rank_id(), dtype=tf.int32), id_uniq_u%get_rank_size()))
            id_uniq_local = tf.gather_nd(id_uniq_u, id_uniq_local_mask)
            id_uniq_local_ = tf.reshape(id_uniq_local, [-1, 1])
            id_uniq_owner = id_uniq_local // get_rank_size()
            huifeng_debug()
            indices_local = tf.reshape(id_uniq_owner, [-1])
            indices_global = tf.reshape(id_uniq_local_mask, [-1])
            #'''
            #indices_local = tf.reshape(model.id_uniq_owner, [-1])
            #indices_global = tf.reshape(model.id_uniq_local_mask, [-1])
            shape_w = tf.shape(model.local_w)
            shape_v = tf.shape(model.local_v)
            embed_values_uniq = _grads[-2] #tf.gradients(model.loss, model.embedding_uniq_)[0]
            w_values_uniq = _grads[-1] #tf.gradients(model.loss, model.W_uniq_)[0]
            
            # fix shape
            grad_embed_unique = math_ops.unsorted_segment_sum(embed_values_uniq, id_uniq_inver, model.fix_shape)
            grad_w_unique = math_ops.unsorted_segment_sum(w_values_uniq, id_uniq_inver, model.fix_shape)
            # dynamic shape
            #grad_embed_unique = math_ops.unsorted_segment_sum(embed_values_uniq, model.id_uniq_inver, array_ops.shape(model.id_uniq_u)[0])
            #grad_w_unique = math_ops.unsorted_segment_sum(w_values_uniq, model.id_uniq_inver, array_ops.shape(model.id_uniq_u)[0])

            #grad_embed_unique = math_ops.unsorted_segment_sum(tf.gradients(model.loss, model.embedding_uniq_)[0], model.id_uniq_inver, array_ops.shape(model.id_uniq_u)[0])
            #grad_w_unique = math_ops.unsorted_segment_sum(tf.gradients(model.loss, model.W_uniq_)[0], model.id_uniq_inver, array_ops.shape(model.id_uniq_u)[0])

            grad_embed_unique = hccl_ops.allreduce(grad_embed_unique,  reduction='sum')#average = average)
            grad_embed_unique/= get_rank_size()
            grad_w_unique = hccl_ops.allreduce(grad_w_unique, reduction='sum')# average=True)
            grad_w_unique/= get_rank_size()
            grad_embed_me = tf.gather(grad_embed_unique, indices_global) 
            grad_w_me = tf.reshape(tf.gather(grad_w_unique, indices_global), [-1,1] )
            grad =  [(tf.IndexedSlices(grad_embed_me, indices_local, shape_v), model.local_v),
                      (tf.IndexedSlices(grad_w_me, indices_local, shape_w), model.local_w)]
            #train_op.append(grad)
            train_op.append(opt[0].apply_gradients([*dense_grads], global_step=global_step))            
            train_op.append(opt[0].apply_gradients([*grad], global_step=global_step))
    return model, train_op

def need_init_varlist(model):
    
    var_list = [var for var in tf.global_variables() if var is not model.W_uniq_ 
            and var is not  model.embedding_uniq_ ] # and var.name is not "h5_w"]

    return var_list

 
def train_dense_ingraph(sess, model, train_op, data_dense, data_unique):
    #step 1 initialize unique embedding and W
    #fetch = [model.W_uniq_.initializer, model.embedding_uniq_.initializer]
    #feed_dict = {model.id_hldr_uniq_all:data_unique[1], model.batch_indices:data_unique[2]}
    #sess.run(fetches=fetch, feed_dict=feed_dict)
    #sess.run(tf.variables_initializer([model.W_uniq_, model.embedding_uniq_]), feed_dict=feed_dict)
    #step 2 get loss and update dense grad input dense data
    #train_op include grad_embed_unique, grad_w_unique, allreuceandapply dense grad apply sparse grad
    #fetch = [model.W_uniq_.initializer, model.embedding_uniq_.initializer, train_op[2], train_op[3]]
    fetch = train_op[0:2]
    fetch += [model.loss, model.log_loss, model.l2_loss, model.train_preds, model.lbl_hldr]
    feed_dict = {model.id_hldr_uniq_all:data_unique[1], model.batch_indices:data_unique[2],
            model.id_hldr_dense: data_dense[1], model.wt_hldr_dense: data_dense[2],
            model.lbl_hldr:data_dense[0],model.wt_hldr_uniq: data_unique[0]
            #model.id_uniq_inver_ori:data_unique[2], model.id_uniq_u:data_unique[1], model.id_uniq_local:data_unique[3],
            # model.wt_hldr_uniq: data_unique[2],
            }
    _, _,  _loss_, _log_loss_, _l2_loss_, _preds_, _train_labels = sess.run(fetches=fetch, feed_dict=feed_dict)
    #print(_mask,"mask",hvd.rank())
    #print(uniq_mask,"range()",hvd.rank())
    #time.sleep()
    #print(_str)
    #huifeng_debug()
    _train_preds = [_preds_.flatten()]
    return  _loss_, _log_loss_, _l2_loss_, _train_preds, _train_labels



def evaluate_general_batch_ingraph(sess, model, batch_ids, batch_ws):
    
    data_comm_me = [ batch_ids[:,:config.common_dim], batch_ws[:,:config.common_dim]]
    data_gen_me = [ batch_ids[:,config.common_dim:], batch_ws[:,config.common_dim:]]
    
    wt_gen_me = batch_ws[:, config.common_dim:]
    id_gen_all = batch_ids[:, config.common_dim:]
    fetch = []
    feed_dict = {model.id_eval_hldr_dense: data_comm_me[0], model.wt_eval_hldr_dense: data_comm_me[1], 
                 model.eval_id_hldr_uniq_all:data_gen_me[0] ,model.wt_eval_hldr_uniq:data_gen_me[1]}

    _preds_ = sess.run(fetches=model.eval_preds, feed_dict=feed_dict)
    batch_preds = [_preds_.flatten()]

    return batch_preds


def evaluate(sess, model, data_dir): # id_hldr, wt_hldr, eval_preds):
    preds = []
    labels = []
    line_cnt =0
    start_time = time.time()
    epoch_finished = False
    test_dataset = input_fn_tfrecord(data_dir, config.test_tag, batch_size * get_rank_size() / config.line_per_sample)
    test_iterator = test_dataset.make_one_shot_iterator()

    next_element = test_iterator.get_next()
    test_num = 0
    while not epoch_finished and test_num<1e10:
        # test_ids, test_wts, test_labels, epoch_finished = test_gen.next()
        test_num += 1
        #print('test num', test_num)
        try:
            test_batch_features = sess.run(next_element)
            test_ids = test_batch_features['feat_ids'].reshape((-1, config.num_inputs))
            test_wts = test_batch_features['feat_vals'].reshape((-1, config.num_inputs))
            test_labels = test_batch_features['label'].reshape((-1, ))

            line_cnt += test_labels.shape[0]
            preds.append(np.squeeze(evaluate_general_batch_ingraph(sess, model, test_ids, test_wts)))
            labels.append(np.squeeze(test_labels))
        except tf.errors.OutOfRangeError:
            print("end of test trainset")
            epoch_finished = True

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    print("evaluate time: %f sec" % (time.time() - start_time))
    return labels,  preds


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    parser.add_argument('--data_dir', default='/autotest/data',
                        help="""directory of dataset.""")

    parser.add_argument('--max_epochs', default=100, type=int,
                        help="""total epochs for training""")

    parser.add_argument('--display_every', default=100, type=int,
                        help="""the frequency to display info""")

    parser.add_argument('--batch_size', default=10000, type=int,
                        help="""batch size for one NPU""")

    parser.add_argument('--max_steps', default=100, type=int,
                        help="""max train steps""")

    # model file
    parser.add_argument('--model_dir', default='./model',
                        help="""model and log directory""")

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    return args


if __name__ == '__main__':

    args = parse_args()
    
    #hvd.init()
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()

    input_dim = config.num_features
    num_inputs = config.num_inputs
    print("input_dim={}, num_inputs={}".format(input_dim, num_inputs))

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    log_file = os.path.join(args.model_dir, algo)
    print("log file: ", log_file)

    batch_size = args.batch_size
    train_per_epoch = config.train_size
    test_per_epoch = config.test_size
    eval_size = args.batch_size

    metric_best = 0
    metric_best_epoch = -1
    #optimizer_array = ['adam', 1e-4, 1e-8, 'mean']
    #optimizer_array = [['adam', 1e-4, 1e-8, 0.5, 5],["ftrl", 0.1, 1, 1e-8, 1e-8]]
    #optimizer_array = [['adam', 5e-4*get_rank_size(), 5e-8, 0.6, 5]]

    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["use_off_line"].b = True  
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  

    ###ge.variableMemoryMaxSize  e.graphMemoryMaxSize,
    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(15401209344))
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(16474836480))
    sess_config.gpu_options.allow_growth = True

    # *** 
    init_sess = tf.Session(config =sess_config)
    init_sess.run(npu_int)


    sess_config.gpu_options.visible_device_list = str(get_local_rank_id())
    global_start_time = time.time()
    
    size = get_rank_size()
    rank = get_rank_id()
    
    #tf.random.set_random_seed(1234)
    with tf.device('/cpu:0'):

        train_dataset = input_fn_tfrecord(args.data_dir, config.train_tag, batch_size=size*args.batch_size / config.line_per_sample,
                                          perform_shuffle=True, num_epochs=args.max_epochs)
        #iterator = train_dataset.make_one_shot_iterator()

        if  get_rank_size() > 1:
            rank_size = get_rank_size()
            rank_id = get_rank_id()
            print("ranksize = %d, rankid = %d" % (rank_size, rank_id))
            train_dataset = train_dataset.shard(rank_size, rank_id)

        iterator = train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        input_data = [tf.reshape(next_element['label'], [-1]),
                      tf.reshape(next_element['feat_ids'], [-1, 39]),
                      tf.reshape(next_element['feat_vals'], [-1, 39])]

        input_data[1] = tf.cast(input_data[1], dtype=tf.int32)

    optimizer_array = [['adam', 5e-4*size, 5e-8, 0.6, 5]]
    with tf.Session(config=sess_config) as sess:
   
        
        model, opt = build_graph_ingraph(optimizer_array, input_data)
       
        writer = tf.summary.FileWriter(args.model_dir, sess.graph)

        #np.ndarray
        feed_dict = {model.id_hldr_uniq_all: np.zeros(shape=(args.batch_size, config.general_dim), dtype=int),
                     model.batch_indices: np.zeros(shape=(args.batch_size), dtype=int)}
        sess.run(iterator.initializer, feed_dict=feed_dict)
        #sess.run(npu_int)
        sess.run(tf.variables_initializer(need_init_varlist(model)), feed_dict=feed_dict)
        #init=tf.global_variables_initializer()
        #sess.run(init,feed_dict=feed_dict) 


        print('model initialized')

        if mode == 'train':

            total_start_time = time.time()

            est_epoch_batches = (train_per_epoch + args.batch_size * size - 1) // (args.batch_size * size)
            print("est_epoch_batches=======", est_epoch_batches)

            _epoch = 1
            train_finished = False
            step = 0
            while _epoch < args.max_epochs + 1 and not train_finished:
                _epoch_start_time = time.time()

                epoch_finished = False
                epoch_finished_batches = 0

                # every batch
                while not epoch_finished and not train_finished:
                    try:

                        input_data_ = sess.run(input_data)

                        batch_size = input_data_[0].shape[0]
                        data_comm_me = [input_data_[0][rank*batch_size:(rank+1)*batch_size],
                                     input_data_[1][rank*batch_size:(rank+1)*batch_size,:config.common_dim],
                                     input_data_[2][rank*batch_size:(rank+1)*batch_size,:config.common_dim]]
                        
                        wt_gen_me = input_data_[2][rank*batch_size:(rank+1)*batch_size, config.common_dim:]
                        id_gen_me = input_data_[1][rank*batch_size:(rank+1)*batch_size, config.common_dim:]
                        batch_indices = np.arange(rank*batch_size, (rank+1)*batch_size)

                        start_time = time.time()
                        _loss, _log_loss, _l2_loss, p, _labels = train_dense_ingraph(sess, model, opt, data_comm_me, 
                                [wt_gen_me, input_data_[1][:,config.common_dim:], batch_indices])
                        end_time = time.time()

                        epoch_finished_batches += 1
                        step +=1

                        step_time = end_time - start_time
                        fps = size * args.batch_size / step_time

                        if step % args.display_every == 0:
                            print('step: %2d, epoch: %3d/%3d, loss: %f, fps: %f, step_time: %f' % (
                                step, _epoch, args.max_epochs, _loss, fps, step_time))

                        if epoch_finished_batches % est_epoch_batches == 0 or epoch_finished:
                            epoch_finished = True
                            print('epoch: %d, train time: %.3f sec' % (_epoch, time.time() - _epoch_start_time))

                            eval_auc = -1
                            epoch_labels = []
                            epoch_preds = []
                            # ======== comment this block if no testset available ========
                            print("== starting evaluate == ")
                            eval_labels, eval_preds = evaluate(sess, model, args.data_dir)
                            eval_auc = metric(log_file, eval_labels, eval_preds)
                            print("current_auc: {}, current_epoch: {} ".format(eval_auc, _epoch))
                            print("== finished evaluate == ")
                            # ============================================================
                            print('epoch: %d, total time: %.3f sec' % (_epoch, time.time() - _epoch_start_time))
                        
                            _epoch += 1

                        if step == args.max_steps:
                            train_finished = True
                            break

                    except tf.errors.OutOfRangeError as e:
                        print("end of training dataset")
                        print("epoch %3d finished ..." % _epoch)
                        train_finished = True

            writer.close()

            print("training {} steps, consume time: {} ".format(step, time.time() - total_start_time))


        elif mode == 'test':
            pass

        tf.train.write_graph(sess.graph, args.model_dir, 'deepfm_graph.pbtxt', as_text=True)
        sess.run(npu_shutdown)
        sess.close()
