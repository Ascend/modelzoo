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

import random
import pathlib
import tensorflow as tf
from utils import paired_random_crop, augment
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image, ext):
    """
    Normalize image to [-1, 1]
    """
    assert ext in ['.png', '.jpg', '.jpeg', '.JPEG']
    if ext == '.png':
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image -= 0.5
    image /= 0.5
    # print(image)

    return image


def load_and_preprocess_image(image_path, ext):
    image = tf.read_file(image_path)
    return preprocess_image(image, ext)


def get_sorted_image_path(path, ext):
    ext_regex = "*" + ext
    data_root = pathlib.Path(path)
    image_paths = list(data_root.glob(ext_regex))
    image_paths = sorted([str(path) for path in image_paths])
    return image_paths


def get_dataset(lr_path, hr_path, ext):
    lr_sorted_paths = get_sorted_image_path(lr_path, ext)
    hr_sorted_paths = get_sorted_image_path(hr_path, ext)

    lr_hr_sorted_paths = list(zip(lr_sorted_paths[:], hr_sorted_paths[:]))
    random.shuffle(lr_hr_sorted_paths)
    lr_sorted_paths, hr_sorted_paths = zip(*lr_hr_sorted_paths)

    ds = tf.data.Dataset.from_tensor_slices(
        (list(lr_sorted_paths), list(hr_sorted_paths)))

    def load_and_preprocess_lr_hr_images(lr_path, hr_path, ext=ext):
        return load_and_preprocess_image(lr_path,
                                         ext), load_and_preprocess_image(
                                             hr_path, ext)

    lr_hr_ds = ds.map(load_and_preprocess_lr_hr_images, num_parallel_calls=16)
    return lr_hr_ds, len(lr_sorted_paths)


def crop_and_augment(lr_hr_ds, FLAGS):

    # def random_crop(lr_img, hr_img, crop_size=FLAGS.crop_size, scale_SR=FLAGS.scale_SR):
    #     hr_img, lr_img = paired_random_crop(hr_img, lr_img, crop_size, scale_SR)
    #     return lr_img, hr_img

    # def data_augmentation(lr_img, hr_img):
    #     hr_img, lr_img = augment([hr_img, lr_img])
    #     return lr_img, hr_img

    def crop_and_aug(lr_img,
                     hr_img,
                     crop_size=FLAGS.crop_size,
                     scale_SR=FLAGS.scale_SR):
        hr_img, lr_img = paired_random_crop(hr_img, lr_img, crop_size,
                                            scale_SR)
        hr_img, lr_img = augment([hr_img, lr_img])
        return lr_img, hr_img

    # random_crop_py_func = lambda lr, hr: tf.py_func(random_crop, [lr, hr], [tf.float32, tf.float32])
    # data_augmentation_py_func = lambda lr, hr: tf.py_func(data_augmentation, [lr, hr], [tf.float32, tf.float32])
    crop_and_aug_py_func = lambda lr, hr: tf.py_func(crop_and_aug, [lr, hr],
                                                     [tf.float32, tf.float32])
    # lr_hr_ds = lr_hr_ds.map(random_crop_py_func, num_parallel_calls=AUTOTUNE)
    # lr_hr_ds = lr_hr_ds.map(data_augmentation_py_func, num_parallel_calls=AUTOTUNE)
    lr_hr_ds = lr_hr_ds.map(crop_and_aug_py_func, num_parallel_calls=AUTOTUNE)
    return lr_hr_ds


def load_train_dataset(lr_path, hr_path, ext, batch_size, FLAGS):
    lr_hr_ds, n_data = get_dataset(lr_path, hr_path, ext)
    lr_hr_ds = crop_and_augment(lr_hr_ds, FLAGS)
    lr_hr_ds = lr_hr_ds.batch(batch_size)
    lr_hr_ds = lr_hr_ds.repeat()
    lr_hr_ds = lr_hr_ds.make_one_shot_iterator()
    return lr_hr_ds, n_data


def load_test_dataset(lr_path, hr_path, ext, batch_size):
    val_lr_hr_ds, val_n_data = get_dataset(lr_path, hr_path, ext)
    val_lr_hr_ds = val_lr_hr_ds.batch(batch_size)
    val_lr_hr_ds = val_lr_hr_ds.repeat()
    return val_lr_hr_ds, val_n_data
