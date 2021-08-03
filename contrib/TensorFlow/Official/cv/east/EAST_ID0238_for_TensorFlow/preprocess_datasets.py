# Copyright 2020 Huawei Technologies Co., Ltd
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

import icdar
import os
import sys
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 20, '')
tf.app.flags.DEFINE_integer('num_readers', 30, '')
tf.app.flags.DEFINE_string('processed_data', './processed_dataset/', 'where to save preprocessed datasets')
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                     input_size=FLAGS.input_size,
                                     batch_size=FLAGS.batch_size_per_gpu)

    if not os.path.isdir(FLAGS.processed_data):
        os.mkdir(FLAGS.processed_data)

    for i in range(1000):
        print("start to proces index {}".format(i))
        data = next(data_generator)
        input_images = np.array(data[0])
        input_score_maps = np.array(data[2])
        input_geo_maps = np.array(data[3])
        input_training_masks = np.array(data[4])

        #save to files
        input_images.tofile(os.path.join(FLAGS.processed_data,'input_images_{}.bin'.format(i)))
        input_score_maps.tofile(os.path.join(FLAGS.processed_data,'input_score_maps_{}.bin'.format(i)))
        input_geo_maps.tofile(os.path.join(FLAGS.processed_data, 'input_geo_maps_{}.bin'.format(i)))
        input_training_masks.tofile(os.path.join(FLAGS.processed_data, 'input_training_masks_{}.bin'.format(i)))