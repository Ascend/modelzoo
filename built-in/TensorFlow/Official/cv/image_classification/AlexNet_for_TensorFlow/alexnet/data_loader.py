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
from tensorflow.python.data.experimental.ops import threadpool
from . import preprocessing
import tensorflow as tf
from tensorflow.python.util import nest
import os,sys
import numpy as np 

class DataLoader:

    def __init__(self, config):
        self.config = config   

        filename_pattern = os.path.join(config.data_dir, '%s-*')
        filenames_train = sorted(tf.gfile.Glob(filename_pattern % 'train'))
        self.num_training_samples = get_num_records(filenames_train)
        self.config.num_training_samples = self.num_training_samples

        filename_pattern = os.path.join(config.data_dir, '%s-*')
        filenames_val = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
        self.num_evaluating_samples = get_num_records(filenames_val)
        self.config.num_evaluating_samples = self.num_evaluating_samples
        
        print( 'total num_training_sampels: %d' %  self.num_training_samples )
        print( 'total num_evaluating_sampels: %d' %  self.num_evaluating_samples )
        
        self.training_samples_per_rank = self.num_training_samples
        
    def get_train_input_fn(self):
        take_count = self.training_samples_per_rank
        batch_size = self.config.batch_size
        shard = self.config.shard

        return make_dataset(self.config, take_count, batch_size,
                    shard=shard,synthetic=self.config.synthetic)

    def get_eval_input_fn(self):
        take_count = self.num_evaluating_samples
        batch_size = self.config.batch_size
        shard = self.config.shard

        return make_dataset_eval(self.config, take_count, batch_size, shard=shard)


#-------------------------------- Funcs -----------------------------------
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

# since the preprocessing is done here, we add config file
def parse_record(raw_record, is_training, cfg):
    image_buffer, label, bbox = _parse_example_proto(raw_record)

    image = preprocessing.parse_and_preprocess_image_record(image_buffer, bbox, training=is_training)
    # label -1 for VGG in ImageNet
    return image, label-1



def make_dataset_eval(config, take_count, batch_size,
                 shard=False, synthetic=False):

    rank_size = int(os.getenv('RANK_SIZE'))
    rank_id = int(os.getenv('DEVICE_INDEX'))

    filename_pattern = os.path.join(config.data_dir, '%s-*')
    filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))

    ds = tf.data.TFRecordDataset.list_files(filenames, shuffle=False)

    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=10, block_length=1)
    if shard:
        ds = ds.shard(rank_size, rank_id)

    
    ds = ds.take(take_count)  # make sure all ranks have the same amount

    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))

    ds = ds.map(lambda image, counter: parse_record(image, False, config), num_parallel_calls=64)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    ds = threadpool.override_threadpool(ds,threadpool.PrivateThreadPool(128, display_name='input_pipeline_thread_pool'))
    return ds


def make_dataset(config, take_count, batch_size,
                 shard=False, synthetic=False):
    shuffle_buffer_size = 10000

    rank_size = int(os.getenv('RANK_SIZE'))
    rank_id = int(os.getenv('RANK_ID'))

    if synthetic:
        height = 224
        width = 224
        input_shape = [batch_size, height, width, 3]
        input_element = nest.map_structure(lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape))
        label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([batch_size, 1]))
        element = (input_element, label_element)
        ds = tf.data.Dataset.from_tensors(element).repeat()
        return ds


    filename_pattern = os.path.join(config.data_dir, '%s-*')
    filenames = tf.gfile.Glob(filename_pattern % 'train')

    ds = tf.data.TFRecordDataset.list_files(filenames, shuffle=False)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=10, block_length=1,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shard:
        ds = ds.shard(rank_size, rank_id)

    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))

    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))

    ds = ds.map(lambda image, counter: parse_record(image, True, config), num_parallel_calls=216)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    ds = threadpool.override_threadpool(ds,threadpool.PrivateThreadPool(128, display_name='input_pipeline_thread_pool'))
    return ds
