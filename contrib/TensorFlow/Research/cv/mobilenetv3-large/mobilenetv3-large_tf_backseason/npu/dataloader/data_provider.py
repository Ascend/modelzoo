# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Functions to read, decode and pre-process input data for the Model.
"""
import collections
import sys
import tensorflow as tf

from tensorflow.python.data.experimental.ops import threadpool

# from tensorflow.contrib import slim

InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'images_orig', 'labels', 'labels_one_hot'])
ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])

DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)


def get_data_files(data_sources):
    from tensorflow.python.platform import gfile
    if isinstance(data_sources, (list, tuple)):
        data_files = []
        for source in data_sources:
            data_files += get_data_files(source)
    else:
        if '*' in data_sources or '?' in data_sources or '[' in data_sources:
            data_files = gfile.Glob(data_sources)
        else:
            data_files = [data_sources]
    if not data_files:
        raise ValueError('No data files found in %s' % (data_sources,))
    return data_files


def preprocess_image(image, location, label_one_hot, height=224, width=224):
    """Prepare one image for evaluation.
    If height and width are specified it would output an image with that size by
    applying resize_bilinear.
    If central_fraction is specified it would cropt the central fraction of the
    input image.
    Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
    Returns:
    3-D float Tensor of prepared image.
    """

    # if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    # if central_fraction:
    #  image = tf.image.central_crop(image, central_fraction=central_fraction)

    # if height and width:
    # Resize the image to the specified height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])

    # image = tf.cast(image, tf.float32)
    # image = tf.multiply(image, 1/255.)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image, location, label_one_hot


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def parse_example_proto(example_serialized, num_classes, labels_offset, image_preprocessing_fn):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
    }
    with tf.compat.v1.name_scope('deserialize_image_record'):
        obj = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
        image = tf.image.decode_jpeg(obj['image/encoded'], channels=3, fancy_upscaling=False,
                                                                 dct_method='INTEGER_FAST')
        if image_preprocessing_fn:
            image = image_preprocessing_fn(image, 224, 224)
        else:
            image = tf.image.resize(image, [224, 224])

        label = tf.cast(obj['image/class/label'], tf.int32)
        label = tf.squeeze(label)
        label -= labels_offset
        label = tf.one_hot(label, num_classes - labels_offset)
        return image, label


def parse_example_decode(example_serialized):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
    }
    with tf.compat.v1.name_scope('deserialize_image_record'):
        obj = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
        image = tf.image.decode_jpeg(obj['image/encoded'], channels=3, fancy_upscaling=False,
                                                                 dct_method='INTEGER_FAST')

    return image, obj['image/class/label']


def parse_example(image, label, num_classes, labels_offset, image_preprocessing_fn):
    with tf.compat.v1.name_scope('deserialize_image_record'):
        if image_preprocessing_fn:
            image = image_preprocessing_fn(image, 224, 224)
        else:
            image = tf.image.resize(image, [224, 224])

        label = tf.cast(label, tf.int32)
        label = tf.squeeze(label)
        label -= labels_offset
        label = tf.one_hot(label, num_classes - labels_offset)
    return image, label


def parse_example1(example_serialized, image_preprocessing_fn1):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
    }
    with tf.compat.v1.name_scope('deserialize_image_record'):
        obj = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
        image = tf.image.decode_jpeg(obj['image/encoded'], channels=3, fancy_upscaling=False,
                                                                 dct_method='INTEGER_FAST')

        image = image_preprocessing_fn1(image, 224, 224)
    return image, obj['image/class/label']


def parse_example2(image, label, num_classes, labels_offset, image_preprocessing_fn2):
    with tf.compat.v1.name_scope('deserialize_image_record'):
        image = image_preprocessing_fn2(image, 224, 224)

        label = tf.cast(label, tf.int32)
        label = tf.squeeze(label)
        label -= labels_offset
        label = tf.one_hot(label, num_classes - labels_offset)
    return image, label


def get_data(dataset, batch_size, num_classes, labels_offset, is_training, 
    preprocessing_name=None, use_grayscale=None, add_image_summaries=False):
    return get_data_united(dataset, batch_size, num_classes, labels_offset, is_training,
        preprocessing_name, use_grayscale, add_image_summaries)


def create_ds(data_sources, is_training):
    data_files = get_data_files(data_sources)
    ds = tf.data.Dataset.from_tensor_slices(data_files)

    if is_training:
        ds = ds.shuffle(1000)
    # add for eval
    else:
        ds = ds.take(50000)

    ##### change #####
    num_readers = 10
    ds = ds.interleave(
        tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))
    ##### change #####

    if is_training:
        ds = ds.repeat()

    return ds


def get_data_united(dataset, batch_size, num_classes, labels_offset, is_training,
    preprocessing_name=None, use_grayscale=None, add_image_summaries=False):
    from preprocessing import preprocessing_factory
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        name='inception_v2',
        is_training=is_training,
        use_grayscale=use_grayscale,
        add_image_summaries=add_image_summaries
    )

    ds = create_ds(dataset.data_sources, is_training)

    ds = ds.map(lambda example, counter: parse_example_proto(example, num_classes, labels_offset, image_preprocessing_fn), num_parallel_calls=24)

    ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    iterator = ds.make_initializable_iterator()

    ds = threadpool.override_threadpool(ds,threadpool.PrivateThreadPool(128, display_name='input_pipeline_thread_pool'))

    return iterator, ds

