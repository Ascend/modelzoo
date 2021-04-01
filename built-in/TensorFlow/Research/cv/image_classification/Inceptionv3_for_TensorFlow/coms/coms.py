
from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt

def load_weights(self):
    pass

def train_model(self):
    pass

def fine_tune(self):
    pass

def predict(self):
    pass

def optimizer_bn(lr, loss, mom=0.9, fun='mm'):
    with tf.name_scope('optimzer_bn'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print('BN parameters: ', update_ops)
        with tf.control_dependencies([tf.group(*update_ops)]):
            optim = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9))
            train_op = slim.learning.create_train_op(loss, optim)
    return train_op

def optimizer(lr, loss, mom=0.9, fun='mm'):
    with tf.name_scope('optimizer'):
        if (fun == 'mm'):
            optim = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9))
        elif (fun == 'gdo'):
            optim = npu_tf_optimizer(tf.train.GradientDescentOptimizer(learning_rate=lr))
        elif (fun == 'adam'):
            optim = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=lr))
        else:
            raise TypeError('未输入正确训练函数')
    return optim

def loss(logits, labels, fun='cross'):
    with tf.name_scope('loss') as scope:
        if (fun == 'cross'):
            _loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            mean_loss = tf.reduce_mean(_loss)
        else:
            raise TypeError('未输入正确误差函数')
        tf.summary.scalar((scope + 'mean_loss'), mean_loss)
    return mean_loss
'\n准确率计算,评估模型\n由于测试存在切片测试合并问题，因此正确率的记录放到了正式代码中\n'

def evaluation(logits, labels):
    with tf.name_scope('evaluation') as scope:
        correct_pre = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accurary = tf.reduce_mean(tf.cast(correct_pre, 'float'))
        tf.summary.scalar((scope + 'accuracy:'), accurary)
    return accurary
if (__name__ == '__main__'):
    pass
