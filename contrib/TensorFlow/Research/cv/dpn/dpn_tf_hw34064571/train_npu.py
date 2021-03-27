import tensorflow as tf
import os
import numpy as np
import DPN_model

import tensorflow.contrib.slim as slim

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
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            mask = tf.image.flip_left_right(mask)
        return image, label, mask

    def _preprocess(image, label, mask):
        image = image - [122.67891434, 116.66876762, 104.00698793]
        image = image / 255.
        return image, label, mask

    dataset = tf.data.TFRecordDataset(tf_file)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, masks_batch = iterator.get_next()
    return images_batch, labels_batch, masks_batch


def training_op(log, label, mask):
    def focal_loss(logits, label, mask, gamma=2):
        epsilon = 1.e-10
        label = tf.one_hot(label, 21)
        probs = tf.nn.softmax(logits)
        probs = tf.clip_by_value(probs, epsilon, 1.)
        gamma_weight = tf.multiply(label, tf.pow(tf.subtract(1., probs), gamma))  # 对正类加gamma系数
        loss = -label * tf.log(probs) * gamma_weight
        loss = tf.reduce_sum(loss, axis=-1)
        loss = loss * mask
        loss = tf.reduce_mean(loss)
        return loss

    loss = focal_loss(log, label, mask)
    optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate_base)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     op = optimizer.minimize(loss, global_step=global_step)
    #     return op, loss
    loss_scaling = 2 ** 15
    grads = optimizer.compute_gradients(loss * loss_scaling)
    # grads = [(grad / loss_scaling, var) for grad, var in grads]
    for i, (grad,var) in enumerate(grads):
        if grad is not None:
            grads[i] = (grad/loss_scaling,var)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads)
        return train_op, loss

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
    "image_num", 10582,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
	"batch_size", 8,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
    "epoch", 60,
    "The config json file corresponding to the pre-trained RESNET model. ")



epochs = FLAGS.epoch
batch_size = FLAGS.batch_size
img_N = FLAGS.image_num
is_training = True
learning_rate_base = 0.0001
_HEIGHT = 512
_WIDTH = 512

pre_model = FLAGS.model_save
train_tf = FLAGS.data_path
model_save = "./model_save/dpn.ckpt"

images_batch, labels_batch, masks_batch = read_data(train_tf, is_training)
print("----------------------------------------------------------------------",images_batch,labels_batch)

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
    train_op, train_loss = training_op(out, inputy, inputm)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, pre_model)
    spectral_distance = Spectral_distance()
    print("Training start....")
    try:
        for epoch in range(epochs):
            confusion_matrix = 0
            for step in range(int(img_N//batch_size)):
                x_in, y_in, m_in = sess.run([images_batch, labels_batch, masks_batch])
                # print('fetch data complete')
                # y_in[y_in == 255] = 0
                _, tra_out, tra_loss = sess.run([train_op, out, train_loss],
                                                    feed_dict={inputx: x_in, inputy: y_in, inputm: m_in,
                                                               inputd: spectral_distance})
                confusion_matrix += evaluating_cm(tra_out, y_in, num_classes=21)
                if (step + 1) % 10 == 0:
                    MIoU = evaluating_miou(confusion_matrix)
                    confusion_matrix = 0
                    print(
                        'Epoch%d,step%d,loss=%.4f,miou=%.4f' % (epoch + 1 , step + 1, tra_loss, MIoU))
                if (step + 1) % 100 == 0:
                    saver.save(sess, model_save, global_step=epoch + 1 )
    except tf.errors.OutOfRangeError:
        print('epoch limit reached')
    finally:
        print("Training Done")
