# coding=utf-8
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
#from hccl.manage.api import get_rank_size
#from hccl.manage.api import get_rank_id

SHUFFLE_BUFFER_SIZE = 10000
NUM_FILES_READ_IN_PARALLEL = 10
NUM_PARALLEL_CALLS = 8
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
IMAGE_SIZE = 224  # this will be used for training and evaluation
MIN_DIMENSION = 256  # when evaluating, resize to this size before doing central crop


class Pipeline:
    def __init__(self, filenames, is_training, batch_size, num_epochs):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size, num_epochs: integers.
        """
        self.is_training = is_training

        # read the files in parallel
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)
        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=NUM_FILES_READ_IN_PARALLEL
        ))
        dataset = dataset.prefetch(buffer_size=batch_size)
        
        # mix the training examples
        if is_training:

            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        dataset = dataset.repeat(num_epochs)
        
        # decode and augment data (对数据进行相关变换)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            self.parse_and_preprocess, batch_size=batch_size,
            num_parallel_batches=1, drop_remainder=True
        
        ))
        
        dataset = dataset.prefetch(buffer_size=1)
        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. Possibly augments it.

        Returns:
            image: a float tensor with shape [height, width, 3],
                a RGB image with pixel values in the range [0, 1].
            label: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image as a string, it will be decoded later
        image_as_string = parsed_features['image']

        # get a label
        label = tf.to_int32(parsed_features['label'])

        if self.is_training:

            # get groundtruth boxes, they must be in from-zero-to-one format,
            # also, it assumed that ymin < ymax and xmin < xmax
            boxes = tf.stack([
                parsed_features['ymin'], parsed_features['xmin'],
                parsed_features['ymax'], parsed_features['xmax']
            ], axis=1)
            boxes = tf.to_float(boxes)  # shape [num_boxes, 4]
            # they are only used for data augmentation

            image = self.augmentation(image_as_string, boxes)
        else:
            image = tf.image.decode_jpeg(image_as_string, channels=3)
            image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range
            image = resize_keeping_aspect_ratio(image, MIN_DIMENSION)
            image = central_crop(image, crop_height=IMAGE_SIZE, crop_width=IMAGE_SIZE)

        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

        # in the format required by tf.estimator,
        # they will be batched later
        features = {'images': image}
        labels = {'labels': label}
        return features, labels

    def augmentation(self, image_as_string, boxes):

        image = get_random_crop(image_as_string, boxes)  # 随机裁剪变换
        image = tf.image.random_flip_left_right(image)  # 随机水平或左右翻转变换
        image = tf.image.resize_images(               # resize变换
            image, [IMAGE_SIZE, IMAGE_SIZE],
            method=RESIZE_METHOD
        )

        image = (1.0 / 255.0) * tf.to_float(image)  # to [0, 1] range# 转换至0-1之间的浮点类型变换
        image = random_color_manipulations(image, probability=0.25, grayscale_probability=0.05)
        return image


def resize_keeping_aspect_ratio(image, min_dimension):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an int tensor with shape [].
    Returns:
        a float tensor with shape [new_height, new_width, 3],
            where min_dimension = min(new_height, new_width).
    """
    image_shape = tf.shape(image)
    height = tf.to_float(image_shape[0])
    width = tf.to_float(image_shape[1])

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension) / original_min_dim
    new_height = tf.round(height * scale_factor)
    new_width = tf.round(width * scale_factor)

    new_size = [tf.to_int32(new_height), tf.to_int32(new_width)]
    image = tf.image.resize_images(image, new_size, method=RESIZE_METHOD)
    return image


def get_random_crop(image_as_string, boxes):

    distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_as_string),
        bounding_boxes=tf.expand_dims(boxes, axis=0),
        min_object_covered=0.25,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )
    # begin:[offset_height,offset_width,0]
    # size:[target_height,target_width,-1]
    begin, size, _ = distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)

    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    crop = tf.image.decode_and_crop_jpeg(
        image_as_string, crop_window, channels=3
    )
    return crop


def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    # 计算裁剪的距离crop_top,crop_left
    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

    # 随机颜色操纵
def random_color_manipulations(image, probability=0.1, grayscale_probability=0.1, fast=True):

    def manipulate(image):
        if not fast:
            # intensity and order of this operations are kinda random,
            # so you will need to tune this for you problem
            image = tf.image.random_brightness(image, 0.15)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.15)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.clip_by_value(image, 0.0, 1.0)
        else:
            image = distort_color_fast(image)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image
    # 以下两个操作是否执行是有生成的随机数大小来决定的
    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)
        # tf.cond有些类似if...else结构
    with tf.name_scope('to_grayscale'):
        make_gray = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(make_gray, lambda: to_grayscale(image), lambda: image)
    return image

    # 快速扭曲颜色
def distort_color_fast(image):
    with tf.name_scope('distort_color'):
        br_delta = tf.random_uniform([], -32.0/255.0, 32.0/255.0)
        cb_factor = tf.random_uniform([], -0.1, 0.1)
        cr_factor = tf.random_uniform([], -0.1, 0.1)
        # 按通道分离
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        red_offset = 1.402 * cr_factor + br_delta
        green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
        blue_offset = 1.772 * cb_factor + br_delta
        # 各通道加上之前的偏移值完成变换
        channels[0] += red_offset
        channels[1] += green_offset
        channels[2] += blue_offset
        # 将通道从新拼接
        image = tf.concat(axis=2, values=channels)
        # 把每一个元素都压缩在min和max之间
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

