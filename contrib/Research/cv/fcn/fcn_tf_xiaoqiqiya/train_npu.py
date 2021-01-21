import tensorflow as tf
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import fcn_pretrain
import tensorflow.contrib.slim as slim
import random


from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def read_data(tf_file, is_training):
    def _parse_read(tfrecord_file):
        features = {
            'image':
                tf.io.FixedLenFeature((), tf.string),
            "label":
                tf.io.FixedLenFeature((), tf.string),
            "mask":
                tf.io.FixedLenFeature((), tf.string),
            'height':
                tf.io.FixedLenFeature((), tf.int64),
            'width':
                tf.io.FixedLenFeature((), tf.int64),
            'channels':
                tf.io.FixedLenFeature((), tf.int64)
        }
        parsed = tf.io.parse_single_example(tfrecord_file, features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
        label = tf.decode_raw(parsed['label'], tf.uint8)
        label = tf.reshape(label, [parsed['height'], parsed['width'], 1])
        mask = tf.decode_raw(parsed['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed['height'], parsed['width'], 1])
        # pad (h,w) to at least 512
        h_pad = tf.clip_by_value(512 - parsed['height'], 0, 100000)
        w_pad = tf.clip_by_value(512 - parsed['width'], 0, 100000)
        image_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0))
        label_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0))
        mask_padding = ((h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0))
        image = tf.pad(image, image_padding, mode='constant', constant_values=0)
        label = tf.pad(label, label_padding, mode='constant', constant_values=0)
        mask = tf.pad(mask, mask_padding, mode='constant', constant_values=0)
        # random crop
        combined = tf.concat([image, label, mask], axis=-1)
        combined = tf.random_crop(combined, (320, 320, 5))
        image = combined[:, :, 0:3]
        label = combined[:, :, 3:4]
        mask = combined[:, :, 4:5]
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int64)
        mask = tf.cast(mask, tf.float32)
        return image, label, mask

    def _augmentation(image, label, mask):
        # 随机翻转
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            mask = tf.image.flip_left_right(mask)
        # 随机亮度变换
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_brightness(image, 32)
        # 随机饱和度变换
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_saturation(image, 0.5, 1.5)
        # 随机对比度变换
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_contrast(image, 0.5, 1.5)
        return image, label, mask

    def _preprocess(image, label, mask):
        image = image / 255.
        image = image - [122.67891434 / 255, 116.66876762 / 255, 104.00698793 / 255]
        return image, label[:, :, 0], mask[:, :, 0]

    dataset = tf.data.TFRecordDataset(tf_file)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    if is_training:
        dataset = dataset.map(_augmentation, num_parallel_calls=2)
        dataset = dataset.map(_preprocess, num_parallel_calls=2)
        dataset = dataset.shuffle(batch_size * 100)
        dataset = dataset.repeat()
    else:
        dataset = dataset.map(_preprocess, num_parallel_calls=2)
        dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, masks_batch = iterator.get_next()
    print(images_batch, labels_batch, masks_batch)
    return images_batch, labels_batch, masks_batch


def check_data(images_batch, labels_batch, gpu):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            for i in range(100):
                x_in, y_in = sess.run([images_batch, labels_batch])
                print(x_in.shape, y_in.shape)
                print(np.max(x_in), np.min(x_in))
                print(np.unique(y_in))


def isin(a, s):
    for i in a:
        if i in s:
            return True
    return False


def training_op(log, label, mask):
    def focal_loss(logits, label, mask, gamma=2):
        epsilon = 1.e-10
        label = tf.one_hot(label, 21)
        probs = tf.nn.softmax(logits)
        probs = tf.clip_by_value(probs, epsilon, 1.)
        gamma_weight = tf.multiply(label, tf.pow(tf.subtract(1., probs), gamma))  # 对正类加gamma系数
        loss = -label * tf.log(probs) * gamma_weight
        print("-----------------------------", loss)
        loss = tf.reduce_sum(loss, axis=-1)
        print("-----------------------------", loss)
        loss = loss * mask
        print("-----------------------------", loss)
        loss = tf.reduce_mean(loss)
        return loss

    loss = focal_loss(log, label, mask)
    optimizer = tf.train.GradientDescentOptimizer(0.00001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op = optimizer.minimize(loss)
        return op, loss


def evaluating_op(log, label, num_classes=21):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    # pixel accuracy
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    # mean iou
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    # mean pixel accuracy
    Mean_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    Mean_acc = np.nanmean(Mean_acc)
    # frequncey iou
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return Pixel_acc, Mean_acc, MIoU, FWIoU




flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "The config json file corresponding to the pre-trained RESNET model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_path", None,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
    "epoch", 30,
    "The config json file corresponding to the pre-trained RESNET model. ")






epochs = FLAGS.epoch
batch_size = 16
img_N = 8498
is_training = True
model=FLAGS.model_path
train_tf = FLAGS.data_path
save_model_path = "./model_save"

images_batch, labels_batch, masks_batch = read_data(train_tf, is_training)
print("-------------------------------", images_batch)
print("-------------------------------", labels_batch)
inputx = tf.placeholder(tf.float32, shape=[batch_size, 320, 320, 3], name="inputx")
inputy = tf.placeholder(tf.int64, shape=[batch_size, 320, 320], name="inputy")
inputm = tf.placeholder(tf.float32, shape=[batch_size, 320, 320], name="inputm")
config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训�?
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开�?
with tf.Session(config=config) as sess:
    with slim.arg_scope(fcn_pretrain.vgg_arg_scope()):
        out = fcn_pretrain.fcn8s(inputx, is_training=is_training)
    train_op, train_loss = training_op(out, inputy, inputm)
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess,model)
  
    print("Training start....")
    try:
        for epoch in range(epochs):
            for step in range(int(img_N / batch_size)):
                x_in, y_in, m_in = sess.run([images_batch, labels_batch, masks_batch])
                m_in = np.where(m_in == 1, 0, 1)
                _, tra_out, tra_loss = sess.run([train_op, out, train_loss],
                                                feed_dict={inputx: x_in, inputy: y_in, inputm: m_in})
                p_acc, m_acc, miou, _ = evaluating_op(tra_out, y_in, num_classes=21)
                # pixel_acc,mean_acc,miou,fwiou = evaluating_op(tra_out,y_in)
                # print(np.array(tra_out).shape)
                if (step + 1) % 10 == 0:
                    print(
                        'Epoch %d, step %d, train loss = %.4f, pixel accuracy= %.2f, mean accuracy= %.2f, miou = %.2f' % (
                            epoch + 1, step + 1, tra_loss, p_acc, m_acc, miou))
            checkpoint_path = os.path.join(save_model_path, "model.ckpt")
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
    except tf.errors.OutOfRangeError:
        print('epoch limit reached')
    finally:
        print("Training Done")
