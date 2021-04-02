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

import argparse
import glob
import math
import os
import random
import time

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--pb_path', type=str, default='./pb_model/unpairedsr.pb', help='Path for pb model file')
parser.add_argument('--data_dir', type=str, default='../data/test', help='Directory for storing the dataset')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_data(npy_path):
    """Load data from the given path

    The data is saved in an npy file, which should be loaded by `np.load` function.
    The loaded data is stored as Python's `dict` type, where the key 'sample' corresponds to low-resolution images,
    and the key 'label' corresponds to high-resolution images.

    Args:
        npy_path: The path of npy data file.

    Returns:
        low-resolution and high-resolution data, both of which are of type ndarray.
        sample_array.shape: [data_size, 16, 16, 3]
        label_array.shape: [data_size, 64, 64, 3]

    """
    data = np.load(npy_path, allow_pickle=True)
    data_dict = data.item()
    sample_imgs = data_dict['sample']
    label_imgs = data_dict['label']

    sample_array = np.array(sample_imgs, dtype=np.float32)
    label_array = np.array(label_imgs, dtype=np.float32)

    sample_array = np.clip(sample_array / 255., 0., 1.)
    label_array = np.clip(label_array / 255., 0., 1.)

    return sample_array, label_array


def get_next(batch_id, batch_size, sample_array, label_array):
    """Get a batch of data

    Args:
        batch_id: The number of the current batch.
        batch_size: The size of batch data.
        sample_array : low-resolution data of shape [data_size, 16, 16, 3]
        label_array: high-resolution data of shape [data_size, 64, 64, 3]

    Returns:
        Batch data and next batch's id.

    """
    last_index = batch_id + batch_size
    batch_sample = sample_array[batch_id: last_index]
    batch_label = label_array[batch_id: last_index]
    return batch_sample, batch_label, last_index


def get_psnr(img1, img2):
    """ Calculate PSNR for a pair of images

    Args:
        img1: An `ndarray` of shape [H, W, C], and its values has been normalized to the interval [0, 1]
        img2: An `ndarray` with the same shape as img1, and its values has been normalized to the interval [0, 1]

    Returns:
        PSNR of the given image pair.
    """
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10 * math.log10(1. / mse)
    return psnr


def main():
    """

    Evaluate PSNR and running time.

    """
    # Loading test dataset
    test_npy_paths = glob.glob(os.path.join(args.data_dir, '*.npy'))
    test_npy_paths.sort()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=sess_config)

    with tf.gfile.FastGFile(args.pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    inputs = tf.get_default_graph().get_tensor_by_name('input:0')
    outputs = tf.get_default_graph().get_tensor_by_name('output:0')

    sum_psnr, tot_time, cnt = 0, 0, 0
    batch_size = 1
    npy_paths = test_npy_paths
    for npy, npy_path in enumerate(npy_paths):
        sample_array, label_array = get_data(npy_path)
        batch_num = math.floor(sample_array.shape[0] / batch_size)
        batch_id = 0
        for step in range(1, int(batch_num) + 1):
            lr_img, hr_img, batch_id = get_next(batch_id, batch_size, sample_array, label_array)
            feed_dict = {inputs: lr_img}
            start_time = time.time()
            fake_hr = sess.run(outputs, feed_dict)
            end_time = time.time()
            sum_psnr += get_psnr(fake_hr, hr_img)
            tot_time += end_time - start_time
            cnt += 1
    avg_psnr = sum_psnr / cnt
    avg_time = tot_time / cnt
    print('avg psnr: %.4f' % avg_psnr)
    print('avg running time: %.2f ms' % (avg_time * 1000))


if __name__ == '__main__':
    # Fix random seed to get stable results
    random_seed = 0
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    # Set TensorFlow's log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)

    main()
