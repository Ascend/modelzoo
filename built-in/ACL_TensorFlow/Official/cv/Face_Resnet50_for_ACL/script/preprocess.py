# MIT License
# 
# Copyright (c) 2018 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
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

import sys
import os
import numpy as np
from scipy import misc
import imp
import time
import math
import random
from datetime import datetime
import shutil
import argparse
import tensorflow as tf
import numpy as np
from collections import namedtuple
StandardFold = namedtuple('StandardFold', ['indices1', 'indices2', 'labels'])
class Dataset():

    def __init__(self, path=None):
        self.num_classes = None
        self.classes = None
        self.images = None
        self.labels = None
        self.features = None
        self.index_queue = None
        self.queue_idx = None
        self.batch_queue = None
        self.is_typeB = None

        if path is not None:
            print(path)
            self.init_from_path(path)

    def init_from_path(self, path):
        print(path)
        path = os.path.expanduser(path)
        _, ext = os.path.splitext(path)
        if os.path.isdir(path):
            self.init_from_folder(path)
        elif ext == '.txt':
            self.init_from_list(path)
        else:
            raise ValueError('Cannot initialize dataset from path: %s\n\
                It should be either a folder or a .txt list file' % path)
        print('%d images of %d classes loaded' % (len(self.images), self.num_classes))

    def init_from_folder(self, folder):
        folder = os.path.expanduser(folder)
        class_names = os.listdir(folder)
        class_names.sort()
        classes = []
        images = []
        labels = []
        for label, class_name in enumerate(class_names):
            classdir = os.path.join(folder, class_name)
            if os.path.isdir(classdir):
                images_class = os.listdir(classdir)
                images_class.sort()
                images_class = [os.path.join(classdir, img) for img in images_class]
                indices_class = np.arange(len(images), len(images) + len(images_class))
                classes.append(DataClass(class_name, indices_class, label))
                images.extend(images_class)
                labels.extend(len(images_class) * [label])
        self.classes = np.array(classes, dtype=np.object)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = len(classes)

    def init_from_list(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        assert len(lines) > 0, \
            'List file must be in format: "fullpath(str) label(int)"'
        images = [line[0] for line in lines]
        if len(lines[0]) > 1:
            labels = [int(line[1]) for line in lines]
        else:
            labels = [os.path.dirname(img) for img in images]
            _, labels = np.unique(labels, return_inverse=True)
        self.images = np.array(images, dtype=np.object)
        self.labels = np.array(labels, dtype=np.int32)
        self.init_classes()

    def init_classes(self):
        dict_classes = {}
        classes = []
        for i, label in enumerate(self.labels):
            if not label in dict_classes:
                dict_classes[label] = [i]
            else:
                dict_classes[label].append(i)
        for label, indices in dict_classes.items():
            classes.append(DataClass(str(label), indices, label))
        self.classes = np.array(classes, dtype=np.object)
        self.num_classes = len(classes)


class DataClass():
    def __init__(self, class_name, indices, label):
        self.class_name = class_name
        self.indices = list(indices)
        self.label = label
        return


class LFWTest:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.images = None
        self.standard_folds = None
        self.queue_idx = None

    def init_standard_proto(self, lfw_pairs_file):
        index_dict = {}
        for i, image_path in enumerate(self.image_paths):
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            index_dict[image_name] = i

        pairs = []
        with open(lfw_pairs_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        # 10 folds
        self.standard_folds = []
        for i in range(10):
            indices1 = np.zeros(600, dtype=np.int32)
            indices2 = np.zeros(600, dtype=np.int32)
            labels = np.array([True] * 300 + [False] * 300, dtype=np.bool)
            # 300 positive pairs, 300 negative pairs in order
            for j in range(600):
                pair = pairs[600 * i + j]
                if j < 300:
                    assert len(pair) == 3
                    img1 = pair[0] + '_' + '%04d' % int(pair[1])
                    img2 = pair[0] + '_' + '%04d' % int(pair[2])
                else:
                    assert len(pair) == 4
                    img1 = pair[0] + '_' + '%04d' % int(pair[1])
                    img2 = pair[2] + '_' + '%04d' % int(pair[3])
                indices1[j] = index_dict[img1]
                indices2[j] = index_dict[img2]
            fold = StandardFold(indices1, indices2, labels)
            self.standard_folds.append(fold)

    def test_standard_proto(self, features):

        assert self.standard_folds is not None

        accuracies = np.zeros(10, dtype=np.float32)
        thresholds = np.zeros(10, dtype=np.float32)

        features1 = []
        features2 = []

        for i in range(10):
            # Training
            train_indices1 = np.concatenate([self.standard_folds[j].indices1 for j in range(10) if j != i])
            train_indices2 = np.concatenate([self.standard_folds[j].indices2 for j in range(10) if j != i])
            train_labels = np.concatenate([self.standard_folds[j].labels for j in range(10) if j != i])

            train_features1 = features[train_indices1, :]
            train_features2 = features[train_indices2, :]

            train_score = - np.sum(np.square(train_features1 - train_features2), axis=1)
            # train_score = np.sum(train_features1 * train_features2, axis=1)
            _, thresholds[i] = utils.accuracy(train_score, train_labels)

            # Testing
            fold = self.standard_folds[i]
            test_features1 = features[fold.indices1, :]
            test_features2 = features[fold.indices2, :]

            test_score = - np.sum(np.square(test_features1 - test_features2), axis=1)
            # test_score = np.sum(test_features1 * test_features2, axis=1)
            accuracies[i], _ = utils.accuracy(test_score, fold.labels, np.array([thresholds[i]]))

        accuracy = np.mean(accuracies)
        threshold = - np.mean(thresholds)
        return accuracy, threshold


def preprocess(images, config, is_training=False):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (config.channels == 1 or config.channels == 3)
        mode = 'RGB' if config.channels == 3 else 'I'
        for image_path in image_paths:
            images.append(misc.imread(image_path, mode=mode))
        images = np.stack(images, axis=0)

    # Process images
    f = {
        'resize': resize,
        'random_crop': random_crop,
        'center_crop': center_crop,
        'random_flip': random_flip,
        'standardize': standardize_images,
        'random_downsample': random_downsample,
    }
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        images = f[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images

def extract_feature(images, batch_size, verbose=False):
    num_images = images.shape[0] if type(images) == np.ndarray else len(images)
    #num_features = self.outputs.shape[1]
    #result = np.ndarray((num_images, num_features), dtype=np.float32)
    start_time = time.time()
    for start_idx in range(0, num_images, batch_size):
        if verbose:
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r'
                             % (num_images, start_idx, elapsed_time))
        end_idx = min(num_images, start_idx + batch_size)
        inputs = images[start_idx:end_idx]
        bin_img = args.output_dir + str(end_idx) + ".bin"
        inputs.tofile(bin_img)
    if verbose:
        print('')
    return


def import_file(full_path_to_module, name='module.name'):
    module_obj = imp.load_source(name, full_path_to_module)

    return module_obj

def resize(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)
    images_new = np.ndarray(shape_new, dtype=images.dtype)
    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def get_new_shape(images, size):
    w, h = tuple(size)
    shape = list(images.shape)
    shape[1] = h
    shape[2] = w
    shape = tuple(shape)
    return shape

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new


def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    w, h = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def random_flip(images):
    images_new = images
    flips = np.random.rand(images_new.shape[0])>=0.5
                
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def standardize_images(images, standard):
    if standard == 'mean_scale':
        mean = 127.5
        std = 128.0
    elif standard == 'scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new


def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images
    ratios = min_ratio + (1 - min_ratio) * np.random.rand(images_new.shape[0])

    for i in range(images_new.shape[0]):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i, :h, :w] = misc.imresize(images[i], (h, w))
        images_new[i] = misc.imresize(images_new[i, :h, :w], (_h, _w))

    return images_new


def main(args):
    config_file = args.config_file
    config = import_file(config_file, 'config')

    testset = Dataset(config.test_dataset_path)
    lfwtest = LFWTest(testset.images)
    lfwtest.init_standard_proto(config.lfw_pairs_file)
    lfwtest.images = preprocess(lfwtest.image_paths, config, is_training=False)
    extract_feature(lfwtest.images, config.batch_size)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    parser.add_argument('output_dir', type=str, help='Data preprocessing output.')
    args = parser.parse_args()
    main(args)
