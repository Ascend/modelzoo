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

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal

random_seed = 0


class MODEL(object):
    """
    Model class
    """

    def __init__(self, config):
        self.config = config
        self.lr_img = tf.placeholder(tf.float32, [None, 16, 16, 3], name='input')
        self.hr_img = tf.placeholder(tf.float32, [None, 64, 64, 3], name='label')
        self.is_train = tf.constant(False, dtype=tf.bool)

        self.global_steps_gen = tf.Variable(0, trainable=False)
        self.global_steps_dis = tf.Variable(0, trainable=False)

        self.res_units = [256, 128, 96]
        self.inp_res_units = [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256], [256, 128, 128], [128, 96, 96]]
        self.dis_blocks = [128, 128, 256, 256, 512, 512]

        # Build model
        self.tanh_out, self.fake_hr = self.generator(self.lr_img)  # (0, 1)
        self.fake_out = self.discriminator(self.fake_hr, self.is_train)
        self.real_out = self.discriminator(self.hr_img, self.is_train)
        self.PSNR = tf.reduce_mean(tf.image.psnr(self.fake_hr, self.hr_img, max_val=1.0))

        # Build optimizer
        alpha, beta = self.config.alpha, self.config.beta
        self.mse_loss = tf.reduce_mean(tf.square(self.fake_hr - self.hr_img))

        generator_loss = tf.reduce_mean(tf.square(self.fake_out - tf.ones_like(self.fake_out)))  # MSE
        self.generator_cost = alpha * self.mse_loss + beta * generator_loss

        discrim_cost_real = tf.reduce_mean(tf.square(self.real_out - tf.ones_like(self.real_out)))
        discrim_cost_gen = tf.reduce_mean(tf.square(self.fake_out - tf.zeros_like(self.fake_out)))
        self.discrim_cost = beta * (discrim_cost_real + discrim_cost_gen)

        dis_optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)
        gen_optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)

        loss_scaling = 2 ** 12
        grads = gen_optimizer.compute_gradients(self.generator_cost * loss_scaling)
        grads = [(grad / loss_scaling, var) for grad, var in grads]
        dis_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.gen_train_op = gen_optimizer.apply_gradients(grads)
            self.dis_train_op = dis_optimizer.minimize(self.discrim_cost, var_list=dis_params,
                                                       global_step=self.global_steps_dis)
            self.all_train_op = tf.group(self.gen_train_op, self.dis_train_op)

    def generator(self, img_lr):
        """Generator

        Args:
            img_lr: Input image data of shape [batch_size, H, W, C]

        Returns:
            Feature

        """
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            out = tf.layers.conv2d(img_lr, filters=256, kernel_size=3, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))
            for ru in range(len(self.res_units)):
                temp = out
                nunits = self.res_units[ru]
                curr_inp_resu = self.inp_res_units[ru]
                if ru == 0:
                    num_blocks_level = 12
                else:
                    num_blocks_level = 3

                for j in range(num_blocks_level):
                    out = self.basic_block(out, in_planes=curr_inp_resu[j], planes=nunits, is_train=self.is_train)

                if ru == 1:
                    temp = tf.layers.conv2d(temp, filters=128, kernel_size=3, padding='same',
                                            kernel_initializer=RandomNormal(seed=random_seed))
                elif ru == 2:
                    temp = tf.layers.conv2d(temp, filters=96, kernel_size=3, padding='same',
                                            kernel_initializer=RandomNormal(seed=random_seed))
                out = out + temp

                # up sample
                if ru < 2:
                    out = tf.layers.conv2d_transpose(out, filters=nunits, kernel_size=4, strides=2, padding='same',
                                                     kernel_initializer=RandomNormal(seed=random_seed))
                    out = tf.layers.batch_normalization(out, training=self.is_train)
                    out = tf.nn.relu(out)
                    out = tf.layers.conv2d(out, filters=nunits, kernel_size=3, padding='same',
                                           kernel_initializer=RandomNormal(seed=random_seed))
                else:
                    out = tf.layers.conv2d(out, filters=nunits, kernel_size=3, padding='same',
                                           kernel_initializer=RandomNormal(seed=random_seed))

            nunits = self.res_units[-1]
            out = tf.layers.conv2d(out, filters=nunits, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, filters=nunits, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))
            out = tf.nn.relu(out)

            out = tf.layers.conv2d(out, filters=3, kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))
            tanh_out = tf.nn.tanh(out)  # (-1, 1)
            out = tf.clip_by_value((tanh_out + 1) / 2, 0, 1)
            return tanh_out, out

    def basic_block(self, x, in_planes, planes, is_train, downsample=False, upsample=False, nobn=False):
        """Build a basic block

        Args:
            x: Feature of shape [batch_size, H, W, C]
            in_planes: The input's channel
            planes: The output's channel
            is_train: True or False
            downsample: Whether to down sample
            upsample: Whether to up sample
            nobn: Whether to execute batch normalization, nobn=True means no batch normalization

        Returns:
            Output feature

        """
        residual = x
        # ---- 1 conv  up / no up
        if not nobn:
            out = tf.layers.batch_normalization(x, training=is_train)
            out = tf.nn.relu(out)
        else:
            out = tf.nn.relu(x)

        if upsample:
            out = tf.layers.conv2d_transpose(out, filters=planes, kernel_size=4, strides=2,
                                             kernel_initializer=RandomNormal(seed=random_seed))
        else:
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=1, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))

        # ---- 2 conv  down / no down
        if not nobn:
            out = tf.layers.batch_normalization(out, training=is_train)
            out = tf.nn.relu(out)
        else:
            out = tf.nn.relu(out)

        if downsample:
            out = tf.nn.avg_pool2d(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=1, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))
        else:
            out = tf.layers.conv2d(out, filters=planes, kernel_size=3, strides=1, padding='same',
                                   kernel_initializer=RandomNormal(seed=random_seed))

        if not in_planes == planes or downsample or upsample:
            if downsample:
                residual = tf.nn.avg_pool2d(residual, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                residual = tf.layers.conv2d(residual, filters=planes, kernel_size=3, strides=1, padding='same',
                                            kernel_initializer=RandomNormal(seed=random_seed))
            elif upsample:
                residual = tf.layers.conv2d_transpose(residual, filters=planes, kernel_size=4, strides=2,
                                                      kernel_initializer=RandomNormal(seed=random_seed))
            else:
                residual = tf.layers.conv2d(residual, filters=planes, kernel_size=3, strides=1, padding='same',
                                            kernel_initializer=RandomNormal(seed=random_seed))
        out += residual
        return out

    def discriminator(self, img, is_training):
        """Discriminator

        Args:
            img: Image data of shape [batch_size, H, W, C]
            is_training: True or False

        Returns:
            An `ndarray` of shape [batch_size, 1] with values in interval [0, 1]

        """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            input_size = 64
            pool_start = len(self.dis_blocks) - 4 if input_size == 64 else len(self.dis_blocks) - 2
            in_feat = 3
            out = img
            for i in range(len(self.dis_blocks)):
                b_down = bool(i >= pool_start)
                out = self.basic_block(out, in_planes=in_feat, planes=self.dis_blocks[i], is_train=is_training,
                                       downsample=b_down, nobn=True)
                in_feat = self.dis_blocks[i]
            out = tf.reshape(out, (-1, 16 * self.dis_blocks[-1]))
            out = tf.layers.dense(out, units=self.dis_blocks[-1], activation=tf.nn.relu,
                                  kernel_initializer=RandomNormal(seed=random_seed))
            out = tf.layers.dense(out, units=1, activation=tf.nn.sigmoid,
                                  kernel_initializer=RandomNormal(seed=random_seed))
            return out
