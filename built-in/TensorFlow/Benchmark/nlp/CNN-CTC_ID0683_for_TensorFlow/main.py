#
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
# Copyright 2021 Huawei Technologies Co., Ltd
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
#
"""

"""
from npu_bridge.npu_init import *

import datetime
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import cnn_lstm_otc_ocr
import utils
import helper

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    print('loading train data')
    train_feeder = utils.DataIterator(data_dir=train_dir)
    print('size: ', train_feeder.size)

    print('loading validation data')
    val_feeder = utils.DataIterator(data_dir=val_dir)
    print('size: {}\n'.format(val_feeder.size))

    num_train_samples = train_feeder.size  # 100000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # example: 100000/100

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  # example: 10000/100
    shuffle_idx_val = np.random.permutation(num_val_samples)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # ***** npu modify begin *****
    global_config = tf.ConfigProto()
    custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    #custom_op.parameter_map["dynamic_input"].b = 1
    #custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    global_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    global_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    # ***** npu modify end ******
    
    with tf.Session(config=global_config) as sess:
    #with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()

            # the training part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch + 1) % 100 == 0:
                    print('batch', cur_batch, ': time', time.time() - batch_time)
                batch_time = time.time()
                indexs = [shuffle_idx[i % num_train_samples] for i in
                          range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                batch_inputs, _, batch_labels = \
                    train_feeder.input_index_generate_batch(indexs)
                # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                # if summary is needed
                summary_str, batch_cost, step, _ = \
                    sess.run([model.merged_summay, model.cost, model.global_step, model.train_op], feed)
                # calculate the cost
                train_cost += batch_cost * FLAGS.batch_size

                train_writer.add_summary(summary_str, step)

                # save the checkpoint
                if step % FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    # logger.info('save checkpoint at step {0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)

                # train_err += the_err * FLAGS.batch_size
                # do validation
                if step % FLAGS.validation_steps == 0:
                    acc_batch_total = 0
                    lastbatch_err = 0
                    lr = 0
                    for j in range(num_batches_per_epoch_val):
                        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                      range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                        val_inputs, _, val_labels = \
                            val_feeder.input_index_generate_batch(indexs_val)
                        val_feed = {model.inputs: val_inputs,
                                    model.labels: val_labels}

                        dense_decoded, lastbatch_err, lr = \
                            sess.run([model.dense_decoded, model.cost, model.lrn_rate],
                                     val_feed)

                        # print the decode result
                        ori_labels = val_feeder.the_label(indexs_val)
                        acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                         ignore_value=-1, isPrint=True)
                        acc_batch_total += acc

                    accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples

                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                          "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                          "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                     lastbatch_err, time.time() - start_time, lr))


def infer(img_path, mode='infer'):
    # imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
    imgList = helper.load_img_path(img_path)
    print(imgList[:5])

    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    total_steps = len(imgList) / FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        for curr_step in range(total_steps):

            imgs_input = []
            seq_len_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                im = cv2.imread(img, 0).astype(np.float32) / 255.
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

                def get_input_lens(seqs):
                    length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)

                    return seqs, length

                inp, seq_len = get_input_lens(np.array([im]))
                imgs_input.append(im)
                seq_len_input.append(seq_len)

            imgs_input = np.asarray(imgs_input)
            seq_len_input = np.asarray(seq_len_input)
            seq_len_input = np.reshape(seq_len_input, [-1])

            feed = {model.inputs: imgs_input}
            dense_decoded_code = sess.run(model.dense_decoded, feed)

            for item in dense_decoded_code:
                expression = ''

                for i in item:
                    if i == -1:
                        expression += ''
                    else:
                        expression += utils.decode_maps[i]

                decoded_expression.append(expression)

        with open('./result.txt', 'a') as f:
            for code in decoded_expression:
                f.write(code + '\n')


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device('/cpu:0'):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

