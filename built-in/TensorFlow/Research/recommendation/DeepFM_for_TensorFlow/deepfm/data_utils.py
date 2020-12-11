# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import configs.config as config
import tensorflow as tf
import os


def input_fn_tfrecord(tag, batch_size=16,
                      num_epochs=1, num_parallel=16, perform_shuffle=False, line_per_sample=1000):

    def extract_fn(data_record):
        features = {
            # Extract features using the keys set during creation
            'label': tf.FixedLenFeature(shape=(line_per_sample, ), dtype=tf.float32),
            'feat_ids': tf.FixedLenFeature(shape=(config.num_inputs * line_per_sample,), dtype=tf.int64),
            'feat_vals': tf.FixedLenFeature(shape=(config.num_inputs * line_per_sample,), dtype=tf.float32),
        }
        sample = tf.parse_single_example(data_record, features)
        sample['feat_ids'] = tf.cast(sample['feat_ids'], dtype=tf.int32)
        #sample['feat_vals'] = sample['feat_vals']*10
        return sample

    path = config.record_path
    all_files = os.listdir(path)
    files = [os.path.join(path,f) for f in all_files if f.startswith(tag)]
    dataset = tf.data.TFRecordDataset(files).map(extract_fn, num_parallel_calls=num_parallel).batch(int(batch_size), drop_remainder=True).repeat(num_epochs)
    # Randomizes input using a window of batch_size elements (read into memory)
    if perform_shuffle:
        #dataset = dataset.shuffle(int(config.batch_size * 1))
        dataset = dataset.shuffle(100)

    # epochs from blending together.
    return dataset
