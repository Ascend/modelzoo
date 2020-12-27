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

from datetime import datetime
import gc
import logging
import math
import os
import moxing as mox

import tensorflow as tf
from sklearn.utils import shuffle

from ops import load_vgg19_weight
from pretrain_generator import train_pretrain_generator
from train_module import Network, Loss, Optimizer
from utils import create_dirs, log, normalize_images, save_image, load_npz_data, load_and_save_data

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


def set_flags():
    Flags = tf.app.flags

    # About data
    Flags.DEFINE_string('data_url', 's3://esrgan-ascend/npz',
                        'The npz data dir')
    Flags.DEFINE_string('HR_data', '/cache/data/npz/DIV2K/DIV2K/HR_sub',
                        'HR data dict')
    Flags.DEFINE_string('LR_data', '/cache/data/npz/DIV2K/DIV2K/LR_sub',
                        'LR data dict')
    Flags.DEFINE_string('HR_npz_filename', 'HR_image.npz',
                        'the filename of HR image npz file')
    Flags.DEFINE_string('LR_npz_filename', 'LR_image.npz',
                        'the filename of LR image npz file')
    Flags.DEFINE_boolean('save_data', True,
                         'Whether to load and save data as npz file')
    Flags.DEFINE_string('train_url', '/cache/data/npz/train_result',
                        'output directory during training')
    Flags.DEFINE_string('native_data', '/cache/data/npz', 'mox copy dst')
    Flags.DEFINE_string(
        'VGG19_weights',
        '/cache/data/npz/pre_train_checkpoint/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'weights url')
    Flags.DEFINE_integer('crop_size', 192,
                         'the size of crop of training HR images')
    # About Network
    Flags.DEFINE_boolean('use_npz', False, 'Whether to use npz data')
    Flags.DEFINE_integer('scale_SR', 4, 'the scale of super-resolution')
    Flags.DEFINE_integer('num_repeat_RRDB', 23,
                         'The number of repeats of RRDB blocks')
    Flags.DEFINE_float('residual_scaling', 0.2, 'residual scaling parameter')
    Flags.DEFINE_integer('initialization_random_seed', 111,
                         'random seed of networks initialization')
    Flags.DEFINE_string('perceptual_loss', 'pixel-wise',
                        'the part of loss function. "VGG19" or "pixel-wise"')
    Flags.DEFINE_string('gan_loss_type', 'RaGAN',
                        'the type of GAN loss functions. "RaGAN or GAN"')

    # About training
    Flags.DEFINE_integer('num_iter', 500000, 'The number of iterations')
    Flags.DEFINE_integer('batch_size', 16, 'Mini-batch size')
    Flags.DEFINE_integer('channel', 3, 'Number of input/output image channel')
    Flags.DEFINE_boolean('pretrain_generator', True,
                         'Whether to pretrain generator')
    Flags.DEFINE_float('pretrain_learning_rate', 2e-4,
                       'learning rate for pretrain')
    Flags.DEFINE_float('pretrain_lr_decay_step', 250000,
                       'decay by every n iteration')
    Flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
    Flags.DEFINE_float('weight_initialize_scale', 0.1,
                       'scale to multiply after MSRA initialization')
    Flags.DEFINE_integer(
        'HR_image_size', 192,
        'Image width and height of HR image. This flag is valid when crop flag is set to false.'
    )
    Flags.DEFINE_integer(
        'LR_image_size', 48,
        'Image width and height of LR image. This size should be 1/4 of HR_image_size exactly. '
        'This flag is valid when crop flag is set to false.')
    Flags.DEFINE_float('epsilon', 1e-12, 'used in loss function')
    Flags.DEFINE_float('gan_loss_coeff', 0.005, 'used in perceptual loss')
    Flags.DEFINE_float('content_loss_coeff', 0.01, 'used in content loss')

    # About log
    Flags.DEFINE_boolean('logging', True, 'whether to record training log')
    Flags.DEFINE_integer('train_sample_save_freq', 2000,
                         'save samples during training every n iteration')
    Flags.DEFINE_integer('train_ckpt_save_freq', 2000,
                         'save checkpoint during training every n iteration')
    Flags.DEFINE_integer('train_summary_save_freq', 200,
                         'save summary during training every n iteration')
    Flags.DEFINE_string('pre_train_checkpoint_dir',
                        '/cache/data/npz/pre_train_checkpoint',
                        'pre-train checkpoint directory')
    Flags.DEFINE_string('checkpoint_dir', '/cache/data/npz/checkpoint_dir',
                        'checkpoint directory')
    Flags.DEFINE_string('logdir', '/cache/data/npz/log_dir', 'log directory')
    Flags.DEFINE_string('host_pre_train',
                        "s3://esrgan-ascend/data/pre_train_checkpoint",
                        'pre train checkpoint')

    return Flags.FLAGS


def set_logger(FLAGS):
    """set logger for training recording"""
    if FLAGS.logging:
        logfile = '{0}/training_logfile_{1}.log'.format(
            FLAGS.logdir,
            datetime.now().strftime("%Y%m%d_%H%M%S"))
        formatter = '%(levelname)s:%(asctime)s:%(message)s'
        logging.basicConfig(level=logging.INFO,
                            filename=logfile,
                            format=formatter,
                            datefmt='%Y-%m-%d %I:%M:%S')
        return True
    else:
        print('No logging is set')
        return False


def main():
    # set flag
    FLAGS = set_flags()
    print('obs path :', FLAGS.data_url)
    mox.file.copy_parallel(FLAGS.data_url, FLAGS.native_data)
    print("mox copy files finished")

    # set logger
    logflag = set_logger(FLAGS)
    log(logflag, 'Training script start', 'info')

    # pre train
    if FLAGS.pretrain_generator:
        train_pretrain_generator(FLAGS, logflag)
        tf.reset_default_graph()
        return
    else:
        log(
            logflag,
            'Pre-train : Pre-train skips and an existing trained model will be used',
            'info')
    LR_data = tf.placeholder(
        tf.float32,
        shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
        name='LR_input')
    HR_data = tf.placeholder(
        tf.float32,
        shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
        name='HR_input')
    # load data
    if FLAGS.save_data:
        log(logflag, 'Data process : Data processing start', 'info')
        HR_train, LR_train = load_and_save_data(FLAGS, logflag)
        log(logflag,
            'Data process : Data loading and data processing are completed',
            'info')
    else:
        log(logflag, 'Data process : Data loading start', 'info')
        HR_train, LR_train = load_npz_data(FLAGS)
        log(
            logflag,
            'Data process : Loading existing data is completed. {} images are loaded'
            .format(len(HR_train)), 'info')
    # build Generator and Discriminator
    network = Network(FLAGS, LR_data, HR_data)
    gen_out = network.generator()
    dis_out_real, dis_out_fake = network.discriminator(gen_out)

    # build loss function
    loss = Loss()
    gen_loss, dis_loss = loss.gan_loss(FLAGS, HR_data, gen_out, dis_out_real,
                                       dis_out_fake)

    # define optimizers
    global_iter = tf.Variable(0, trainable=False)
    dis_var, dis_optimizer, gen_var, gen_optimizer = Optimizer().gan_optimizer(
        FLAGS, global_iter, dis_loss, gen_loss)

    # build summary writer
    tr_summary = tf.summary.merge(loss.add_summary_writer())

    num_train_data = len(LR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))

    HR_train, LR_train = normalize_images(HR_train, LR_train)

    fetches = {
        'dis_optimizer': dis_optimizer,
        'gen_optimizer': gen_optimizer,
        'dis_loss': dis_loss,
        'gen_loss': gen_loss,
        'gen_HR': gen_out,
        'summary': tr_summary
    }

    gc.collect()

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    # Start Session
    with tf.Session(config=config) as sess:
        log(logflag, 'Training ESRGAN starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)

        writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph)

        pre_saver = tf.train.Saver(var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
        pre_saver.restore(sess,
                          tf.train.latest_checkpoint(FLAGS.host_pre_train))

        if FLAGS.perceptual_loss == 'VGG19':
            sess.run(load_vgg19_weight(FLAGS))
        log(logflag, 'load pretrain vgg19 model', 'info')
        saver = tf.train.Saver(max_to_keep=10)

        for epoch in range(num_epoch):
            log(logflag, 'ESRGAN Epoch: {0}'.format(epoch), 'info')
            HR_train, LR_train = shuffle(HR_train, LR_train, random_state=222)

            for iteration in range(num_batch_in_train):
                current_iter = tf.train.global_step(sess, global_iter)
                if current_iter > FLAGS.num_iter:
                    break

                feed_dict = {
                    HR_data:
                    HR_train[iteration *
                             FLAGS.batch_size:iteration * FLAGS.batch_size +
                             FLAGS.batch_size],
                    LR_data:
                    LR_train[iteration *
                             FLAGS.batch_size:iteration * FLAGS.batch_size +
                             FLAGS.batch_size]
                }

                # update weights of G/D
                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                # save summary every n iter
                if current_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'],
                                       global_step=current_iter)

                # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    log(
                        logflag,
                        'ESRGAN iteration : {0}, gen_loss : {1}, dis_loss : {2}'
                        .format(current_iter, result['gen_loss'],
                                result['dis_loss']), 'info')

                    save_image(FLAGS,
                               result['gen_HR'],
                               'train',
                               current_iter,
                               save_max_num=5)
                # # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    log(
                        logflag,
                        'ESRGAN iteration : {0}, gen_loss : {1}, dis_loss : {2}'
                        .format(current_iter, result['gen_loss'],
                                result['dis_loss']), 'info')
                #
                #     save_image(FLAGS, result['gen_HR'], 'train', current_iter, save_max_num=5)
                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess,
                               os.path.join(FLAGS.checkpoint_dir, 'gen'),
                               global_step=current_iter)

        writer.close()
        mox.file.copy_parallel(FLAGS.checkpoint_dir,
                               "s3://esrgan-ascend/data/checkpoint")
        log(logflag, 'Training ESRGAN end', 'info')
        log(logflag, 'Training script end', 'info')


if __name__ == '__main__':
    main()
    mox.file.copy_parallel("/var/log/npu/slog", "s3://esrgan-ascend/host_log")
