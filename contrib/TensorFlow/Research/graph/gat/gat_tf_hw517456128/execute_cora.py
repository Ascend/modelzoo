# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
import time
import numpy as np
import tensorflow as tf
import logging

from models import GAT
from utils import process

import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# NPU config
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

# training params
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'dataset string')
flags.DEFINE_string('data_url', './data', 'input directory')
flags.DEFINE_string('train_url', './output', 'output directory')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('nb_epochs', 200, 'number of epochs')
flags.DEFINE_integer('patience', 100, 'for early stopping')
flags.DEFINE_float('lr', 0.005, 'learning rate')
flags.DEFINE_float('l2_coef', 0.0005, 'weight decay')
flags.DEFINE_list('hid_units', [8], 'numbers of hidden units per each attention head in each layer')
flags.DEFINE_list('n_heads', [8, 1], 'additional entry for the output layer')
flags.DEFINE_boolean('residual', False, '')

nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + FLAGS.dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(FLAGS.lr))
print('l2_coef: ' + str(FLAGS.l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(FLAGS.hid_units)))
print('nb. units per layer: ' + str(FLAGS.hid_units))
print('nb. attention heads: ' + str(FLAGS.n_heads))
print('residual: ' + str(FLAGS.residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

checkpt_file = FLAGS.train_url + 'mod_cora.ckpt'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(FLAGS.dataset, FLAGS.data_url)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=FLAGS.hid_units, n_heads=FLAGS.n_heads,
                                residual=FLAGS.residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, FLAGS.lr, FLAGS.l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.train_url, "train"), graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=os.path.join(FLAGS.train_url, "test"), graph=sess.graph)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(FLAGS.nb_epochs):
            t = time.time()

            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * FLAGS.batch_size < tr_size:
                _, loss_value_tr, acc_tr, summary = sess.run([train_op, loss, accuracy, summary_op],
                    feed_dict={
                        ftr_in: features[tr_step*FLAGS.batch_size:(tr_step+1)*FLAGS.batch_size],
                        bias_in: biases[tr_step*FLAGS.batch_size:(tr_step+1)*FLAGS.batch_size],
                        lbl_in: y_train[tr_step*FLAGS.batch_size:(tr_step+1)*FLAGS.batch_size],
                        msk_in: train_mask[tr_step*FLAGS.batch_size:(tr_step+1)*FLAGS.batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1
            train_writer.add_summary(summary, epoch)

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * FLAGS.batch_size < vl_size:
                loss_value_vl, acc_vl, summary = sess.run([loss, accuracy, summary_op],
                    feed_dict={
                        ftr_in: features[vl_step*FLAGS.batch_size:(vl_step+1)*FLAGS.batch_size],
                        bias_in: biases[vl_step*FLAGS.batch_size:(vl_step+1)*FLAGS.batch_size],
                        lbl_in: y_val[vl_step*FLAGS.batch_size:(vl_step+1)*FLAGS.batch_size],
                        msk_in: val_mask[vl_step*FLAGS.batch_size:(vl_step+1)*FLAGS.batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            logger = logging.getLogger()
            while logger.handlers:
                logger.handlers.pop()
            tf.logging.info('Epoch %d Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f | Time: %.5f' %
                    (epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step, time.time()-t))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == FLAGS.patience:
                    tf.logging.info('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    tf.logging.info('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * FLAGS.batch_size < ts_size:
            loss_value_ts, acc_ts, summary = sess.run([loss, accuracy, summary_op],
                feed_dict={
                    ftr_in: features[ts_step*FLAGS.batch_size:(ts_step+1)*FLAGS.batch_size],
                    bias_in: biases[ts_step*FLAGS.batch_size:(ts_step+1)*FLAGS.batch_size],
                    lbl_in: y_test[ts_step*FLAGS.batch_size:(ts_step+1)*FLAGS.batch_size],
                    msk_in: test_mask[ts_step*FLAGS.batch_size:(ts_step+1)*FLAGS.batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
        test_writer.add_summary(summary, epoch)

        tf.logging.info('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        train_writer.close()
        test_writer.close()

        sess.close()
