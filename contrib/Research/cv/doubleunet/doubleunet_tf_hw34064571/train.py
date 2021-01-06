import os
import tensorflow as tf
import tensorflow.nn as nn
import numpy as np
from DoubleUnet import doubleunet
from Dataset import VOC2012
import moxing as mox
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import math

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", './dataset/',
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_string(
    "output_path", './model/',
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_bool(
    "is_training", False,
    "Whether to run training.")
flags.DEFINE_integer(
    "class_num", 2,
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_integer(
    "img_num", 2445,
    "train images number")
flags.DEFINE_integer(
    "batch_size", 16,
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_integer(
    "epochs", 300,
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_integer(
    "_HEIGHT", 256,
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_integer(
    "_WIDTH", 320,
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_integer(
    "_DEPTH", 3,
    "The config json file corresponding to the pre-trained DoubleUnet model. ")
flags.DEFINE_float(
    "learning_rate_base", 1e-5,
    "The initial learning rate for GradientDescent.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
DATA_CACHE_PATH = FLAGS.data_path
MODEL_CACHE_PATH = FLAGS.output_path


def pre_process(x, y):
    if np.max(y) > 2:
        y = y / 255
    x = x / 255
    return x, y


def evaluating_op(log, label, num_classes=2):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def plateau_decay(learning_rate, global_step, loss, data_count, batch_size, factor=0.1, patience=10, min_delta=1e-4,
                  cooldown=0, min_lr=0):
    steps_per_epoch = math.ceil(data_count // batch_size)
    patient_steps = patience * steps_per_epoch
    cooldown_steps = cooldown * steps_per_epoch
    if not isinstance(learning_rate, tf.Tensor):
        learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(learning_rate), trainable=False,
                                        collections=[tf.GraphKeys.LOCAL_VARIABLES])

    step = tf.get_variable('step', trainable=False, initializer=global_step,
                           collections=[tf.GraphKeys.LOCAL_VARIABLES])
    best = tf.get_variable('best', trainable=False, initializer=tf.constant(np.Inf, tf.float32),
                           collections=[tf.GraphKeys.LOCAL_VARIABLES])

    def _update_best():
        with tf.control_dependencies([tf.assign(best, loss), tf.assign(step, global_step), ]):
            print('Plateau Decay: Updated Best - Step:', global_step, 'Next Decay Step:',
                  global_step + patient_steps, 'Loss:', loss)
            return tf.identity(learning_rate)

    def _decay():
        with tf.control_dependencies([tf.assign(best, loss),
                                      tf.assign(learning_rate, tf.maximum(tf.multiply(learning_rate, factor), min_lr)),
                                      tf.assign(step, global_step + cooldown_steps), ]):
            print('Plateau Decay: Decayed LR - Step:', global_step, 'Next Decay Step:',
                  global_step + cooldown_steps + patient_steps, 'Learning Rate:', learning_rate)
            return tf.identity(learning_rate)

    def _no_op():
        print('no_op')
        return tf.identity(learning_rate)

    met_threshold = tf.less(loss, best - min_delta)
    should_decay = tf.greater_equal(global_step - step, patient_steps)

    return tf.cond(met_threshold, _update_best, lambda: tf.cond(should_decay, _decay, _no_op))


def main(_):
    voc2012 = VOC2012(image_size=(FLAGS._HEIGHT, FLAGS._WIDTH))
    voc2012.load_all_data(DATA_CACHE_PATH + '/cvcdb_train.h5', DATA_CACHE_PATH + '/cvcdb_val.h5')
    inputx = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS._HEIGHT, FLAGS._WIDTH, FLAGS._DEPTH],
                            name="inputx")
    inputy = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS._HEIGHT, FLAGS._WIDTH], name="inputy")
    out = doubleunet(inputx, is_training=FLAGS.is_training)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=inputy)
    loss = tf.reduce_mean(loss)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = plateau_decay(FLAGS.learning_rate_base, global_step, loss, voc2012.train_images.shape[0],
                                  FLAGS.batch_size)
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=50)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('searching checkpoint')
    ckpt = tf.train.get_checkpoint_state(MODEL_CACHE_PATH)
    pre_epoch = 0
    if ckpt:
        print('loading checkpoint')
        pre_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
        saver.restore(sess, ckpt.model_checkpoint_path)

    if FLAGS.is_training:
        print("Training start....")
        total_step = int(FLAGS.img_num / FLAGS.batch_size)
        print('train dataset shape:' + str(voc2012.train_images.shape[0]))
        for epoch in range(FLAGS.epochs - pre_epoch):
            voc2012.data_shuffle()
            MIoU = 0
            batch_loss = 0
            for step in range(total_step):
                x_in, y_in = voc2012.get_batch_train(batch_size=FLAGS.batch_size)
                x_in, y_in = pre_process(x_in, y_in)
                _, batch_out, batch_loss_ = sess.run([train_op, out, loss], feed_dict={inputx: x_in, inputy: y_in})
                MIoU += evaluating_op(batch_out, y_in)
                batch_loss += batch_loss_
            print('Epoch%d, train loss=%.4f,train_miou=%.4f' % (
                epoch + 1 + pre_epoch, batch_loss / total_step, MIoU / total_step))
            checkpoint_path = os.path.join(MODEL_CACHE_PATH, "model.ckpt")
            saver.save(sess, checkpoint_path, global_step=epoch + pre_epoch + 1)

    else:
        print('start testing')
        MIoU = 0
        total_step = int(voc2012.val_images.maxshape[0] / FLAGS.batch_size) + 1
        for step in range(total_step):
            x_in, y_in = voc2012.get_batch_val(batch_size=FLAGS.batch_size)
            x_in, y_in = pre_process(x_in, y_in)
            mini_out = sess.run(out, feed_dict={inputx: x_in})
            MIoU += evaluating_op(mini_out, y_in)
        print('mean_iou=%.4f' % (MIoU / total_step))


if __name__ == '__main__':
    tf.app.run()
