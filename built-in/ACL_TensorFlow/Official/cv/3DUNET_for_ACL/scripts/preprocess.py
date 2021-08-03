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
import numpy as np
import time
import os
import sys
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from transforms import Cast, CenterCrop, OneHotLabels, NormalizeImages, apply_transforms, PadXYZ


def cross_validation(x: np.ndarray, fold_idx: int, n_folds: int):
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError('Fold index has to be [0, n_folds). Received index {} for {} folds'.format(fold_idx, n_folds))

    _folders = np.array_split(x, n_folds)

    return np.concatenate(_folders[:fold_idx] + _folders[fold_idx + 1:]), _folders[fold_idx]

class Dataset:
    def __init__(self, data_dir, batch_size=1, fold_idx=0, n_folds=5, seed=0, pipeline_factor=1, params=None):
        self._folders = np.array([os.path.join(data_dir, path) for path in os.listdir(data_dir)])
        self._folders.sort()
        self._train, self._eval = cross_validation(self._folders, fold_idx=fold_idx, n_folds=n_folds)
        self._pipeline_factor = pipeline_factor
        self._data_dir = data_dir
        self.params = params

        self._batch_size = batch_size
        self._seed = seed

        self._xshape = (240, 240, 155, 4)
        self._yshape = (240, 240, 155)

    def parse(self, serialized):
        features = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'Y': tf.io.FixedLenFeature([], tf.string),
            'mean': tf.io.FixedLenFeature([4], tf.float32),
            'stdev': tf.io.FixedLenFeature([4], tf.float32)
        }

        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)

        x = tf.io.decode_raw(parsed_example['X'], tf.uint8)
        x = tf.cast(tf.reshape(x, self._xshape), tf.uint8)
        y = tf.io.decode_raw(parsed_example['Y'], tf.uint8)
        y = tf.cast(tf.reshape(y, self._yshape), tf.uint8)

        mean = parsed_example['mean']
        stdev = parsed_example['stdev']

        return x, y, mean, stdev

    def eval_fn(self):
        ds = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        ds = ds.cache()
        ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            OneHotLabels(n_classes=4),
            PadXYZ()
        ]

        ds = ds.map(map_func=lambda x, y, mean, stdev: apply_transforms(x, y, mean, stdev, transforms=transforms),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size=self._batch_size,
                      drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds


if __name__ == '__main__':

    filepath = sys.argv[1]
    output = sys.argv[2]
    label = sys.argv[3]

    dataset = Dataset(data_dir=filepath, batch_size=1, fold_idx=0, n_folds=5)
    ds = dataset.eval_fn()
    iter = ds.make_initializable_iterator()

    ds_sess = tf.Session()
    ds_sess.run(iter.initializer)
    next_element = iter.get_next()
    i = 0
    while True:
        try:
            input = ds_sess.run(next_element)
            batch_data = input[0]
            batch_labels = input[1]
            batch_data.tofile(output + "input_" + str(i) + ".bin")
            batch_labels.tofile(label + "label_" + str(i) +".bin")

            i += 1
        except tf.errors.OutOfRangeError:
            break

    ds_sess.close()

