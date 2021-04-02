
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import tensorflow as tf
import math

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt

class DPN():

    def __init__(self, config, input_shape, num_classes, weight_decay, keep_prob, data_format):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.prob = (1.0 - keep_prob)
        assert (data_format in ['channels_last', 'channels_first'])
        self.data_format = data_format
        self.config = config
        assert (len(config['init_conv_filters']) == len(config['init_conv_kernel_size']) == len(config['init_conv_strides']))
        assert (len(config['first_dpn_block_filters']) == 3)
        assert (config['first_dpn_block_filters'][0] == config['first_dpn_block_filters'][1])
        assert ((config['first_dpn_block_filters'][1] % config['G']) == 0)
        assert (len(config['k']) == len(config['dpn_block_list']))
        self.first_dpn_block_filters = config['first_dpn_block_filters']
        self.k = config['k']
        self.G = config['G']
        self.dpn_block_list = config['dpn_block_list']
        self.block_list_filters = [[(i * config['first_dpn_block_filters'][0]), (i * config['first_dpn_block_filters'][1]), (i * config['first_dpn_block_filters'][2])] for i in range(1, (len(config['dpn_block_list']) + 1))]
        self.global_step = tf.train.get_or_create_global_step()
        self.is_training = True
        self._define_inputs()
        self._build_graph()
        self._init_session()

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.input_shape)
        self.images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):
        with tf.variable_scope('init_conv'):
            init_conv = self.images
            for i in range(len(self.config['init_conv_filters'])):
                init_conv = self._conv_bn_activation(bottom=init_conv, filters=self.config['init_conv_filters'][i], kernel_size=self.config['init_conv_kernel_size'][i], strides=self.config['init_conv_strides'][i])
            pool1 = self._max_pooling(bottom=init_conv, pool_size=self.config['init_pooling_pool_size'], strides=self.config['init_pooling_strides'], name='pool1')
        dpn_block = pool1
        for i in range(len(self.block_list_filters)):
            strides = (1 if (i == 0) else 2)
            dpn_block = self._dpn_block(dpn_block, self.block_list_filters[i], strides, self.k[i], self.dpn_block_list[i], ('dpn_block_' + str((i + 1))))
        bn = self._bn(dpn_block)
        relu = tf.nn.relu(bn)
        with tf.variable_scope('final_dense'):
            axes = ([1, 2] if (self.data_format == 'channels_last') else [2, 3])
            global_pool = tf.reduce_mean(relu, axis=axes, keepdims=False, name='global_pool')
            dropout = self._dropout(global_pool, 'dropout')
            final_dense = tf.layers.dense(dropout, self.num_classes, name='final_dense')
        with tf.variable_scope('optimizer'):
            self.logit = tf.nn.softmax(final_dense, name='logit')
            self.classifer_loss = tf.losses.softmax_cross_entropy(self.labels, final_dense, label_smoothing=0.1, reduction=tf.losses.Reduction.MEAN)
            self.l2_loss = (self.weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]))
            total_loss = (self.classifer_loss + self.l2_loss)
            lossavg = tf.train.ExponentialMovingAverage(0.9, name='loss_moveavg')
            lossavg_op = lossavg.apply([total_loss])
            with tf.control_dependencies([lossavg_op]):
                self.total_loss = tf.identity(total_loss)
            var_list = tf.trainable_variables()
            varavg = tf.train.ExponentialMovingAverage(0.9, name='var_moveavg')
            varavg_op = varavg.apply(var_list)
            optimizer = npu_tf_optimizer(tf.train.MomentumOptimizer(self.lr, momentum=0.9))
            train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group([update_ops, lossavg_op, varavg_op, train_op])
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_dense, 1), tf.argmax(self.labels, 1)), tf.float32), name='accuracy')

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def train_one_batch(self, images, labels, lr, sess=None):
        self.is_training = True
        if (sess is None):
            sess_ = self.sess
        else:
            sess_ = sess
        (_, loss, acc) = sess_.run([self.train_op, self.total_loss, self.accuracy], feed_dict={self.images: images, self.labels: labels, self.lr: lr})
        return (loss, acc)

    def validate_one_batch(self, images, labels, sess=None):
        self.is_training = False
        if (sess is None):
            sess_ = self.sess
        else:
            sess_ = sess
        (logit, acc) = sess_.run([self.logit, self.accuracy], feed_dict={self.images: images, self.labels: labels, self.lr: 0.0})
        return (logit, acc)

    def test_one_batch(self, images, sess=None):
        self.is_training = False
        if (sess is None):
            sess_ = self.sess
        else:
            sess_ = sess
        logit = sess_.run([self.logit], feed_dict={self.images: images, self.lr: 0.0})
        return logit

    def save_weight(self, mode, path, sess=None):
        assert (mode in ['latest', 'best'])
        if (sess is None):
            sess_ = self.sess
        else:
            sess_ = sess
        saver = (self.saver if (mode == 'latest') else self.best_saver)
        saver.save(sess_, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, mode, path, sess=None):
        assert (mode in ['latest', 'best'])
        if (sess is None):
            sess_ = self.sess
        else:
            sess_ = sess
        saver = (self.saver if (mode == 'latest') else self.best_saver)
        ckpt = tf.train.get_checkpoint_state(path)
        if (ckpt and ckpt.model_checkpoint_path):
            saver.restore(sess_, path)
            print('load', mode, 'model in', path, 'successfully')
        else:
            raise FileNotFoundError('Not Found Model File!')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(inputs=bottom, axis=(3 if (self.data_format == 'channels_last') else 1), training=self.is_training)
        return bn

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d(inputs=bottom, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format=self.data_format, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        bn = self._bn(conv)
        if (activation is not None):
            return activation(bn)
        else:
            return bn

    def _bn_activation_conv(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        bn = self._bn(bottom)
        if (activation is not None):
            bn = activation(bn)
        conv = tf.layers.conv2d(inputs=bn, filters=filters, kernel_size=kernel_size, strides=strides, padding='same', data_format=self.data_format, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        return conv

    def _group_conv(self, bottom, filters, kernel_size, strides):
        total_conv = []
        filters_per_path = (filters // self.G)
        for i in range(self.G):
            conv = self._bn_activation_conv(bottom, filters_per_path, kernel_size, strides)
            total_conv.append(conv)
        axes = (3 if (self.data_format == 'channels_last') else 1)
        total_conv = tf.concat(total_conv, axis=axes)
        return total_conv

    def _dpn_block(self, bottom, filters, strides, k, blocks, scope):
        with tf.variable_scope(scope):
            axes = (3 if (self.data_format == 'channels_last') else 1)
            dense_layers = []
            dpn = bottom
            project = self._bn_activation_conv(bottom, (filters[2] + (2 * k)), 1, strides)
            if (self.data_format == 'channels_last'):
                shortcut = project[:, :, :, :filters[2]]
                dense_layers.append(project[:, :, :, filters[2]:])
            else:
                shortcut = project[:, :filters[2], :, :]
                dense_layers.append(project[:, filters[2]:, :, :])
            for i in range(blocks):
                dpn = self._bn_activation_conv(dpn, filters[0], 1, 1)
                dpn = self._group_conv(dpn, filters[1], 3, (strides if (i == 0) else 1))
                dpn = self._bn_activation_conv(dpn, (filters[2] + k), 1, 1)
                if (self.data_format == 'channels_last'):
                    residual = dpn[:, :, :, :filters[2]]
                    dense = dpn[:, :, :, filters[2]:]
                else:
                    residual = dpn[:, :filters[2], :, :]
                    dense = dpn[:, filters[2]:, :, :]
                residual = (residual + shortcut)
                shortcut = residual
                dense_layers.append(dense)
                dpn = tf.concat(([residual] + dense_layers), axis=axes)
            return dpn

    def _max_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.max_pooling2d(inputs=bottom, pool_size=pool_size, strides=strides, padding='same', data_format=self.data_format, name=name)

    def _avg_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.average_pooling2d(inputs=bottom, pool_size=pool_size, strides=strides, padding='same', data_format=self.data_format, name=name)

    def _dropout(self, bottom, name):
        return tf.layers.dropout(inputs=bottom, rate=self.prob, training=self.is_training, name=name)
