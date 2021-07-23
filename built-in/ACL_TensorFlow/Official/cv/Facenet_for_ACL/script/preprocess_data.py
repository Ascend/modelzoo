# MIT License
# 
# Copyright (c) 2016 David Sandberg
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from tensorflow.python.ops import data_flow_ops
from scipy import misc
# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')

def get_control_flag(control, field):
    return tf.equal(tf.math.mod(tf.math.floordiv(control, field), 2), 1)

def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.io.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)
            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),
                            lambda:tf.py_function(random_rotate_image, [image], tf.uint8),
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], RANDOM_CROP),
                            lambda:tf.image.random_crop(image, image_size + (3,)),
                            lambda:tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),
                            lambda:tf.image.random_flip_left_right(image),
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),
                            #lambda:(tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda:(tf.cast(image, tf.float32) - 0)/1,
                            lambda:tf.cast(tf.image.per_image_standardization(image),tf.float32))
            image = tf.cond(get_control_flag(control[0], FLIP),
                            lambda:tf.image.flip_left_right(image),
                            lambda:tf.identity(image))
            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder,
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)

    return image_batch, label_batch
def evaluate(sess,output_path, enqueue_op,image_paths_placeholder,labels_placeholder,control_placeholder,batch_size_placeholder, labels, image_paths, actual_issame, batch_size, use_flipped_images,eval_input_queue,image_batch,use_fixed_image_standardization):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*FIXED_STANDARDIZATION
    if use_flipped_images:
        control_array += (labels_array % 2)*FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    #embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    print("#############nrof_batches",nrof_batches)
    if not os.path.exists(output_path +"data_image_bin/"):
        os.makedirs(output_path +"data_image_bin/")
    if not os.path.exists(output_path +"data_label_bin/"):
        os.makedirs(output_path +"data_label_bin/")
    for i in range(nrof_batches):
        ###########save  bin ###############
        print(i,image_paths[i])
        feed_dict2 = {batch_size_placeholder:batch_size}
        mid_name = image_paths[i].split('.')[0].split('/')[-1]
        bin_image2 = output_path +"data_image_bin/" + mid_name + '_'+str(i)+'_'+".bin"
        bin_label2 = output_path +"data_label_bin/" + mid_name + '_'+str(i)+'_'+".bin"
        emb, lab  = sess.run([image_batch,labels],feed_dict=feed_dict2)
        emb.astype(np.uint8).tofile(bin_image2)
        lab.tofile(bin_label2)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
        ######################
def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            output_path = args.output_dir
            # Read the file containing the pairs used for testing
            pairs = read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = get_paths(os.path.expanduser(args.lfw_dir), pairs)
            print(len(paths))
            print(len(actual_issame))
            #print(paths)
            #print(actual_issame)
            image_paths_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None, 1), name='image_paths')
            labels_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='labels')
            batch_size_placeholder = tf.compat.v1.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='control')
            phase_train_placeholder = tf.compat.v1.placeholder(tf.bool, name='phase_train')

            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                                       dtypes=[tf.string, tf.int32, tf.int32],
                                                       shapes=[(1,), (1,), (1,)],
                                                       shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many(
                [image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads,
                                                                     batch_size_placeholder)

            coord = tf.train.Coordinator()
            threads =tf.train.start_queue_runners(coord=coord, sess=sess)
            evaluate(sess,output_path,eval_enqueue_op, image_paths_placeholder,labels_placeholder,control_placeholder,batch_size_placeholder, label_batch, paths, actual_issame, args.lfw_batch_size,args.use_flipped_images,eval_input_queue,image_batch,args.use_fixed_image_standardization)
            sess.run(eval_input_queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)
def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list
def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('output_dir', type=str, help='Data preprocessing output.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--distance_metric', type=int,
        help='Distance metric  0:euclidian, 1:cosine similarity.', default=0)
    parser.add_argument('--use_flipped_images',
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--subtract_mean',
        help='Subtract feature mean before calculating distance.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization',
        help='Performs fixed standardization of images.', action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
