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
import numpy as np
from . import preprocessing
from . import inception_preprocessing
import tensorflow as tf
from tensorflow.python.util import nest
import os,sys
import numpy as np 
from tensorflow.python.data.experimental.ops import threadpool

IMAGE_SIZE = 224

class DataLoader:

    def __init__(self, args):
        self.args = args   

        filename_pattern = os.path.join(args.data_path, '%s-*')
        filenames_train = sorted(tf.gfile.Glob(filename_pattern % 'train'))
        self.num_training_samples = get_num_records(filenames_train)
        self.args.num_training_samples = self.num_training_samples

        filename_pattern = os.path.join(args.data_path, '%s-*')
        filenames_val = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
        self.num_evaluating_samples = get_num_records(filenames_val)
        self.args.num_evaluating_samples = self.num_evaluating_samples
        
        print( 'total num_training_sampels: %d' %  self.num_training_samples )
        print( 'total num_evaluating_sampels: %d' %  self.num_evaluating_samples )
        
        self.training_samples_per_rank = self.num_training_samples
        
    def get_train_input_fn(self):
        take_count = self.training_samples_per_rank

        return make_dataset(self.args, take_count, self.args.batch_size, training=True)

    def get_eval_input_fn(self):
        take_count = self.num_evaluating_samples

        return make_dataset(self.args, take_count, self.args.batch_size, training=False)


def get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count

    nfile = len(filenames)
    return (count_records(filenames[0]) * (nfile - 1) +
            count_records(filenames[-1]))


def _parse_example_proto(example_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox


# since the preprocessing is done here, we add args file
def parse_record(raw_record, is_training):
    image_buffer, label, bbox = _parse_example_proto(raw_record)
    image = tf.image.decode_jpeg(image_buffer, channels = 3)
    image = inception_preprocessing.preprocess_image(image, IMAGE_SIZE, IMAGE_SIZE, is_training=is_training)

    #image = preprocessing.parse_and_preprocess_image_record(image_buffer, bbox, training=is_training)

    return image, label



def make_dataset(args, take_count, batch_size,
                 training=False, shard=False):

    shuffle_buffer_size = 10000
    num_readers = 10

    rank_size = int(os.getenv('RANK_SIZE'))
    rank_id = int(os.getenv('DEVICE_INDEX'))

    if training:
        filename_pattern = os.path.join(args.data_path, '%s-*')
        filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
    else:
        filename_pattern = os.path.join(args.data_path, '%s-*')
        filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

    ds = tf.data.Dataset.from_tensor_slices(filenames)

    if not training:
        ds = ds.take(take_count)

    if training:
        # ds = ds.shuffle(1024, seed=7*(1+rank_id))
        ds = ds.shuffle(1024)

    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    #counter = tf.data.Dataset.range(sys.maxsize)
    #ds = tf.data.Dataset.zip((ds, counter))

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)
    ds = ds.prefetch(buffer_size = batch_size)

    if training:
        # ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size, seed=5*(1+rank_id)))
        ds = ds.shuffle(buffer_size = shuffle_buffer_size)
        ds = ds.repeat()

    ds = ds.map(lambda image: parse_record(image, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds = ds.map(lambda image, counter: parse_record(image, training), num_parallel_calls=24)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    ds = threadpool.override_threadpool(ds,
                                        threadpool.PrivateThreadPool(128,display_name='input_pipeline_threa_pool'))
    return ds


