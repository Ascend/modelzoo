import tensorflow as tf
import os
import numpy as np
import sys
import DPN_model
import tensorflow.contrib.slim as slim
import random
import time
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["GE_USE_STATIC_MEMORY"]="1"
# os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
os.environ['EXPERIMENTAL_DYNAMIC_PARTITION']='1'
# os.environ["GE_USE_STATIC_MEMORY"]="1"
# os.environ['DUMP_GE_GRAPH'] = '1'
# os.environ['DUMP_GRAPH_LEVEL'] = '1'


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
        label = tf.reshape(label, [parsed['height'], parsed['width'], parsed['channels']])
        mask = tf.decode_raw(parsed['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed['height'], parsed['width'], 24])
        label = label[:, :, 0:1]
        mask = mask[:, :, 0:1]

        image, label, mask = _augmentation(image, label, mask, parsed['height'], parsed['width'])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int64)
        mask = tf.cast(mask, tf.float32)
        image, label, mask = _preprocess(image, label, mask)
        return image, label[:, :, 0], mask[:, :, 0]

    def _augmentation(image, label, mask, h, w):
        image = tf.image.resize_images(image, size=[512, 512], method=0)
        image = tf.cast(image, tf.uint8)
        label = tf.image.resize_images(label, size=[64, 64], method=1)
        mask = tf.image.resize_images(mask, size=[64, 64], method=1)
        # 随机翻转
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            mask = tf.image.flip_left_right(mask)
        return image, label, mask

    def _preprocess(image, label, mask):
        image = image - [122.67891434, 116.66876762, 104.00698793]
        image = image / 255.
        return image, label, mask

    dataset = tf.data.TFRecordDataset(tf_file, num_parallel_reads=2)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size * 10))
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_parse_read, batch_size=batch_size, drop_remainder=True,
                                      num_parallel_calls=2))
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, masks_batch = iterator.get_next()
    return images_batch, labels_batch, masks_batch


def evaluating_cm(log, label, num_classes=21):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    cm = count.reshape(num_classes, num_classes)
    return cm


def evaluating_miou(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def Spectral_distance():
    spectral_distance = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            spectral_distance[i, j] = (i - 5) ** 2 + (j - 5) ** 2
    spectral_distance = (spectral_distance / 50).astype(np.float32)
    spectral_distance = np.expand_dims(spectral_distance, axis=-1)
    return spectral_distance


def isin(a, s):
    for i in a:
        if i in s:
            return True
    return False



flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "The config json file corresponding to the pre-trained RESNET model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_path", None,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_string(
    "output_path", None,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
    "image_num", 1449,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
	"batch_size", 9,
    "The config json file corresponding to the pre-trained RESNET model. ")



batch_size = FLAGS.batch_size
img_N = FLAGS.image_num
is_training = False
_HEIGHT = 512
_WIDTH = 512
pre_model = FLAGS.model_path
val_tf =  FLAGS.data_path

images_batch, labels_batch, masks_batch = read_data(val_tf, is_training)
inputx = tf.placeholder(tf.float32, shape=[batch_size, _HEIGHT, _WIDTH, 3], name="inputx")
inputy = tf.placeholder(tf.int64, shape=[batch_size, _HEIGHT // 8, _WIDTH // 8], name="inputy")
inputm = tf.placeholder(tf.float32, shape=[batch_size, _HEIGHT // 8, _WIDTH // 8], name="inputm")
inputd = tf.placeholder(tf.float32, shape=[10, 10, 1], name="inputd")
config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训�?
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开�?
with tf.Session(config=config) as sess:
    with slim.arg_scope(DPN_model.vgg_arg_scope()):
        out = DPN_model.dpn_model(inputx, inputd, is_training=is_training, _HEIGHT=_HEIGHT, _WIDTH=_WIDTH)
    print('searching checkpoint')
    pre_train_saver = tf.train.Saver()
    pre_train_saver.restore(sess, pre_model)
    print("test start....")
    spectral_distance = Spectral_distance()
    try:
        confusion_matrix = 0
        for step in range(int(img_N / batch_size)):
            x_in, y_in, m_in = sess.run([images_batch, labels_batch, masks_batch])
            print("------------------step----------------------",step)
            tra_out = sess.run(out, feed_dict={inputx: x_in, inputd: spectral_distance})
            confusion_matrix += evaluating_cm(tra_out, y_in, num_classes=21)
        MIoU = evaluating_miou(confusion_matrix)
        print('MIoU:%f' % MIoU)
    except tf.errors.OutOfRangeError:
        print('epoch limit reached')
    finally:
        print("test Done")
