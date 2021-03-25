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

from collections import OrderedDict

import tensorflow as tf

from network import Generator, Discriminator, Perceptual_VGG19


class Network(object):
    """class to build networks"""
    def __init__(self, FLAGS, LR_data=None, HR_data=None):
        self.FLAGS = FLAGS
        self.LR_data = LR_data
        self.HR_data = HR_data

    def generator(self):
        with tf.name_scope('generator'):
            with tf.variable_scope('generator'):
                gen_out = Generator(self.FLAGS).build(self.LR_data)

        return gen_out

    def discriminator(self, gen_out):
        discriminator = Discriminator(self.FLAGS)

        with tf.name_scope('real_discriminator'):
            with tf.variable_scope('discriminator', reuse=False):
                dis_out_real = discriminator.build(self.HR_data)

        with tf.name_scope('fake_discriminator'):
            with tf.variable_scope('discriminator', reuse=True):
                dis_out_fake = discriminator.build(gen_out)

        return dis_out_real, dis_out_fake


class Loss(object):
    """class to build loss functions"""
    def __init__(self):
        self.summary_target = OrderedDict()

    def pretrain_loss(self, pre_gen_out, HR_data):
        with tf.name_scope('loss_function'):
            with tf.variable_scope('pixel-wise_loss'):
                pre_gen_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.abs(pre_gen_out - HR_data),
                                  axis=[1, 2, 3]))

        self.summary_target['pre-train : pixel-wise_loss'] = pre_gen_loss
        return pre_gen_loss

    def _perceptual_vgg19_loss(self, HR_data, gen_out):
        with tf.name_scope('perceptual_vgg19_HR'):
            with tf.variable_scope('perceptual_vgg19', reuse=False):
                vgg_out_hr = Perceptual_VGG19().build(HR_data)

        with tf.name_scope('perceptual_vgg19_Gen'):
            with tf.variable_scope('perceptual_vgg19', reuse=True):
                vgg_out_gen = Perceptual_VGG19().build(gen_out)

        return vgg_out_hr, vgg_out_gen

    def gan_loss(self, FLAGS, HR_data, gen_out, dis_out_real, dis_out_fake):

        with tf.name_scope('loss_function'):
            with tf.variable_scope('loss_generator'):
                if FLAGS.gan_loss_type == 'RaGAN':
                    g_loss_p1 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_real - tf.reduce_mean(dis_out_fake),
                            labels=tf.zeros_like(dis_out_real)))

                    g_loss_p2 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_fake - tf.reduce_mean(dis_out_real),
                            labels=tf.ones_like(dis_out_fake)))

                    gen_loss = FLAGS.gan_loss_coeff * (g_loss_p1 + g_loss_p2)
                elif FLAGS.gan_loss_type == 'GAN':
                    gen_loss = FLAGS.gan_loss_coeff * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_fake,
                            labels=tf.ones_like(dis_out_fake)))
                else:
                    raise ValueError('Unknown GAN loss function type')

                # content loss : L1 distance
                content_loss = FLAGS.content_loss_coeff * tf.reduce_mean(
                    tf.reduce_sum(tf.abs(gen_out - HR_data), axis=[1, 2, 3]))

                gen_loss += content_loss

                # perceptual loss
                if FLAGS.perceptual_loss == 'pixel-wise':
                    perc_loss = tf.reduce_mean(
                        tf.reduce_mean(tf.square(gen_out - HR_data), axis=3))
                    gen_loss += perc_loss
                elif FLAGS.perceptual_loss == 'VGG19':
                    vgg_out_gen, vgg_out_hr = self._perceptual_vgg19_loss(
                        HR_data, gen_out)
                    perc_loss = tf.reduce_mean(
                        tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr),
                                       axis=3))
                    gen_loss += perc_loss
                else:
                    raise ValueError('Unknown perceptual loss type')

            with tf.variable_scope('loss_discriminator'):
                if FLAGS.gan_loss_type == 'RaGAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_real - tf.reduce_mean(dis_out_fake),
                            labels=tf.ones_like(dis_out_real)))

                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_fake - tf.reduce_mean(dis_out_real),
                            labels=tf.zeros_like(dis_out_fake)))

                    dis_loss = d_loss_real + d_loss_fake
                elif FLAGS.gan_loss_type == 'GAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_real,
                            labels=tf.ones_like(dis_out_real)))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=dis_out_fake,
                            labels=tf.zeros_like(dis_out_fake)))

                    dis_loss = d_loss_real + d_loss_fake

                else:
                    raise ValueError('Unknown GAN loss function type')

            self.summary_target['generator_loss'] = gen_loss
            self.summary_target['content_loss'] = content_loss
            self.summary_target['perceptual_loss'] = perc_loss
            self.summary_target['discriminator_loss'] = dis_loss
            self.summary_target['discriminator_real_loss'] = d_loss_real
            self.summary_target['discriminator_fake_loss'] = d_loss_fake

        return gen_loss, dis_loss

    def add_summary_writer(self):
        return [
            tf.summary.scalar(key, value)
            for key, value in self.summary_target.items()
        ]


class Optimizer(object):
    """class to build optimizers"""
    @staticmethod
    def pretrain_optimizer(FLAGS, global_iter, pre_gen_loss):
        learning_rate = tf.train.cosine_decay(FLAGS.pretrain_learning_rate,
                                              global_iter % 250000,
                                              250000,
                                              alpha=1e-7)
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_generator'):
                pre_gen_var = tf.compat.v1.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                pre_gen_optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(
                        loss=pre_gen_loss,
                        global_step=global_iter,
                        var_list=pre_gen_var)

        return pre_gen_var, pre_gen_optimizer

    @staticmethod
    def gan_optimizer(FLAGS, global_iter, dis_loss, gen_loss):
        boundaries = [20000, 40000, 60000, 80000]
        values = [
            FLAGS.learning_rate, FLAGS.learning_rate * 0.5,
            FLAGS.learning_rate * 0.5**2, FLAGS.learning_rate * 0.5**3,
            FLAGS.learning_rate * 0.5**4
        ]
        learning_rate = tf.train.piecewise_constant(global_iter, boundaries,
                                                    values)

        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_discriminator'):
                dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='discriminator')
                dis_optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(loss=dis_loss,
                                                          var_list=dis_var)

            with tf.variable_scope('optimizer_generator'):
                with tf.control_dependencies([dis_optimizer]):
                    gen_var = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    gen_optimizer = tf.train.AdamOptimizer(
                        learning_rate=learning_rate).minimize(
                            loss=gen_loss,
                            global_step=global_iter,
                            var_list=gen_var)
        return dis_var, dis_optimizer, gen_var, gen_optimizer
