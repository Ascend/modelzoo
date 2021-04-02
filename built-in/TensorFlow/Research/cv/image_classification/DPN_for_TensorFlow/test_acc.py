import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import time
import DPN as net
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

config = {
    # dpn92 [96, 96, 256], dpb98 [160, 160, 256]
    'first_dpn_block_filters': [96, 96, 256],
    # dpn92 [3, 4, 20, 3], dpb98 [3, 6, 20, 3]
    'dpn_block_list': [3, 4, 20, 3],

    # parameters for conv and pool before dense block
    'init_conv_filters': [16],
    'init_conv_kernel_size': [3],
    'init_conv_strides': [1],
    'init_pooling_pool_size': 3,
    'init_pooling_strides': 2,

    # dpn92 [16, 32, 24, 128], dpb98 [16, 32, 32, 128]
    'k': [16, 32, 24, 128],
    # dpn92 32, dpb98 40
    'G': 32
}

mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
data_shape = (32, 32, 3)
num_train = 3000
num_test = 10000
num_classes = 10
train_batch_size = 128
test_batch_size = 200
epochs = 10
weight_decay = 1e-4
keep_prob = 0.8
lr = math.sqrt(0.1)

(x_train, y_train) , (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
train_gen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
).flow(x_train, y_train, batch_size=train_batch_size)
test_gen = ImageDataGenerator().flow(x_test, y_test, batch_size=test_batch_size)

reduce_lr_epoch = [epochs//2, 3*epochs//4]
testnet = net.DPN(config, data_shape, num_classes, weight_decay, keep_prob, 'channels_last')
tf.io.write_graph(testnet.sess.graph, 'output', 'graph.pbtxt', as_text=True)
for epoch in range(epochs):
    print('-'*20, 'epoch', epoch, '-'*20)
    train_acc = []
    train_loss = []
    test_acc = []
    # reduce learning rate
    if epoch in reduce_lr_epoch:
        lr = lr * 0.1
        print('reduce learning rate =', lr, 'now')
    begin_epoch = time.time()
    # train one epoch
    for iter in range(num_train//train_batch_size):
        # get and preprocess image
        images, labels = train_gen.next()
        images = images - mean
        begin = time.time()
        # train_one_batch also can accept your own session
        loss, acc = testnet.train_one_batch(images, labels, lr)
        end = time.time()
        train_acc.append(acc)
        train_loss.append(loss)
        print('step:', str(iter + 1) + '/' + str(num_train // train_batch_size), 'time:', '%.3fs' % (end - begin), 'loss:', loss, 'acc:', acc)
        sys.stdout.flush()
    end_epoch = time.time()
    mean_train_loss = np.mean(train_loss)
    mean_train_acc = np.mean(train_acc)
    print('>> epoch:', epoch + 1, 'time:', '%.3fs' % (end_epoch - begin_epoch), 'loss:', mean_train_loss, 'acc:', mean_train_acc)
    sys.stdout.flush()
    tf.train.Saver().save(testnet.sess, "output/model.ckpt-" + str(epoch))

    # logit = testnet.test(images)
    # testnet.save_weight(self, mode, path, sess=None)
    # testnet.load_weight(self, mode, path, sess=None)


