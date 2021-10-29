# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random
import imageio as misc

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def load_tiny() :
    IMAGENET_MEAN = [123.68, 116.78, 103.94]
    path = './tiny-imagenet-200'
    num_classes = 200

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype=np.float32)
    y_train = np.zeros([num_classes * 500], dtype=np.float32)

    trainPath = path + '/train'

    print('loading training images...')

    i = 0
    j = 0
    annotations = {}
    annos = os.listdir(trainPath)
    annos.sort()
    for sChild in annos:
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = misc.imread(os.path.join(sChildPath, c), pilmode='RGB')
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype=np.float32)
    y_test = np.zeros([num_classes * 50], dtype=np.float32)

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    testPics = os.listdir(testPath)
    testPics.sort()
    for sChild in testPics:
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = misc.imread(sChildPath, pilmode='RGB')
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images : ' + str(i))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    X_train, X_test = normalize(X_train, X_test)

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    X_train = np.transpose(X_train, [0, 3, 2, 1])
    X_test = np.transpose(X_test, [0, 3, 2, 1])

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    return X_train, y_train, X_test, y_test

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def get_annotations_map():
    valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch, img_size, dataset_name):
    if dataset_name == 'mnist' :
        batch = _random_crop(batch, [img_size, img_size], 4)

    elif dataset_name =='tiny' :
        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [img_size, img_size], 8)

    else :
        batch = _random_flip_leftright(batch)
        batch = _random_crop(batch, [img_size, img_size], 4)
    return batch

if __name__ == "__main__":
    output_dir = "./input_bins/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_x, train_y, test_x, test_y = load_tiny()

    with open('./imagenet_tiny_label.txt','w') as f:
        for i in range(len(test_x)):
            print('start to process image: {}'.format(i))
            index = i
            image = test_x[i]
            image.tofile(os.path.join(output_dir,'val_{}.bin'.format(index)))
            label = test_y[i]
            f.write('val_{}.JPEG {}\n'.format(index, np.argmax(label)))
