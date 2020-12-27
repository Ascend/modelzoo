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

import gc
import os
import math

from sklearn.utils import shuffle
import tensorflow as tf

from ops import scale_initialization
from train_module import Network, Loss, Optimizer
from utils import log, normalize_images, save_image
import glob
from data import load_train_dataset
import moxing as mox
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


def train_pretrain_generator(FLAGS, logflag):
    """pre-train deep network as initialization weights of ESRGAN Generator"""
    log(logflag, 'Pre-train : Process start', 'info')

    LR_data = tf.placeholder(
        tf.float32,
        shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
        name='LR_input')
    HR_data = tf.placeholder(
        tf.float32,
        shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
        name='HR_input')

    # build Generator
    network = Network(FLAGS, LR_data)
    pre_gen_out = network.generator()

    # build loss function
    loss = Loss()
    pre_gen_loss = loss.pretrain_loss(pre_gen_out, HR_data)

    # build optimizer
    global_iter = tf.Variable(0, trainable=False)
    pre_gen_var, pre_gen_optimizer = Optimizer().pretrain_optimizer(
        FLAGS, global_iter, pre_gen_loss)

    # build summary writer
    pre_summary = tf.summary.merge(loss.add_summary_writer())

    num_train_data = len(glob.glob(FLAGS.LR_data + '/*'))
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))

    fetches = {
        'pre_gen_loss': pre_gen_loss,
        'pre_gen_optimizer': pre_gen_optimizer,
        'gen_HR': pre_gen_out,
        'summary': pre_summary
    }

    gc.collect()

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    saver = tf.train.Saver(max_to_keep=1)

    # Start session
    with tf.Session(config=config) as sess:
        log(logflag, 'Pre-train : Training starts', 'info')
        lr_hr_ds, n_data = load_train_dataset(os.path.join(FLAGS.LR_data),
                                              os.path.join(FLAGS.HR_data),
                                              '.png', FLAGS.batch_size, FLAGS)
        next_element = lr_hr_ds.get_next()
        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, FLAGS))

        writer = tf.summary.FileWriter(FLAGS.logdir,
                                       graph=sess.graph,
                                       filename_suffix='pre-train')

        for epoch in range(num_epoch):
            log(logflag, 'Pre-train Epoch: {0}'.format(epoch), 'info')

            for iteration in range(num_batch_in_train):
                current_iter = tf.train.global_step(sess, global_iter)
                lr, hr = sess.run(next_element)
                if current_iter > FLAGS.num_iter:
                    break
                feed_dict = {LR_data: lr, HR_data: hr}

                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)

                # save summary every n iter
                if current_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'],
                                       global_step=current_iter)

                # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    log(
                        logflag,
                        'Pre-train iteration : {0}, pixel-wise_loss : {1}'.
                        format(current_iter, result['pre_gen_loss']), 'info')

                # save checkpoint
                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess,
                               os.path.join(FLAGS.pre_train_checkpoint_dir,
                                            'pre_gen'),
                               global_step=current_iter)
                    mox.file.copy_parallel(
                        FLAGS.pre_train_checkpoint_dir,
                        "s3://esrgan-ascend/checkpoint/pre_train")
                    mox.file.copy_parallel(FLAGS.logdir,
                                           "s3://esrgan-ascend/root_log")
        writer.close()
        log(logflag, 'Pre-train : Process end', 'info')
