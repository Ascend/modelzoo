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

import os
import glob
import imageio
import numpy as np
import random
import json
import tensorflow as tf


class Minibatch(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            Minibatch.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self,
                 data_dir,
                 set_file='train.json',
                 batch_size=2,
                 num_frames=7,
                 scale=4,
                 in_size=[32, 32],
                 drop_remainder=True,
                 noise_augmenter=None):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.scale = scale
        self.in_size = in_size
        self.drop_remainder = drop_remainder
        self.noise_augmenter = noise_augmenter

        set_file = os.path.join(data_dir, 'sets', set_file)
        with open(set_file, 'r') as fid:
            meta = json.load(fid)

        self.lrcliplist = []
        self.hrcliplist = []
        for vid in meta['videos']:
            if meta['prefix']:
                in_path = os.path.join(data_dir, 'images', meta['x{}_folder'.format(scale)], vid['name'])
                gt_path = os.path.join(data_dir, 'images', meta['gt_folder'], vid['name'])
            else:
                in_path = os.path.join(data_dir, 'images', vid['name'], meta['x{}_folder'.format(scale)])
                gt_path = os.path.join(data_dir, 'images', vid['name'], meta['gt_folder'])
            inList = sorted(glob.glob(os.path.join(in_path, '*.png')))
            gtList = sorted(glob.glob(os.path.join(gt_path, '*.png')))

            vidlen = len(inList)
            assert len(inList) == len(gtList)

            for i in range(0, vidlen - num_frames + 1):
                self.lrcliplist.append(inList[i:i + self.num_frames])
                self.hrcliplist.append(gtList[i + self.num_frames // 2])

        self.total_samples = len(self.lrcliplist)
        if self.total_samples == 0:
            raise FileNotFoundError(f'Found no files in {data_dir}. '
                                    f'Please make sure the data folder structure is correct, '
                                    f'and the meta-data in "{set_file}" is correct.')

        self.index = list(range(self.total_samples))
        random.shuffle(self.index)
        self.cur = 0

    def imread_and_crop(self, im_file):
        im = imageio.imread(im_file)
        h, w, _ = im.shape
        h_offset = random.randint(0, h - self.in_size[0] - 1)
        w_offset = random.randint(0, w - self.in_size[1] - 1)
        return im[h_offset:h_offset + self.in_size[0], w_offset:w_offset + self.in_size[1]]

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next()

    def get_next(self):
        assert self.drop_remainder is True
        if self.cur + self.batch_size > self.total_samples:
            random.shuffle(self.index)
            self.cur = 0

        lrlist = [self.lrcliplist[idx] for idx in self.index[self.cur:self.cur + self.batch_size]]
        hrlist = [self.hrcliplist[idx] for idx in self.index[self.cur:self.cur + self.batch_size]]
        lr = [np.array([imageio.imread(perimg) / 255. for perimg in pervid]).astype(np.float32) for pervid in lrlist]
        hr = [np.array([imageio.imread(pervid) / 255.]).astype(np.float32) for pervid in hrlist]

        lr_crop, hr_crop = [], []
        for ilr, ihr in zip(lr, hr):
            t1, h1, w1, c1 = ilr.shape
            t2, h2, w2, c2 = ihr.shape
            assert c1 == c2 and h1 * self.scale == h2 and w1 * self.scale == w2 and t1 == self.num_frames and t2 == 1
            h_offset = random.randint(0, h1 - self.in_size[0] - 1)
            w_offset = random.randint(0, w1 - self.in_size[1] - 1)
            lr_crop.append(ilr[:, h_offset:h_offset + self.in_size[0], w_offset:w_offset + self.in_size[1]])
            hr_crop.append(ihr[:, h_offset * self.scale:(h_offset + self.in_size[0]) * self.scale, w_offset * self.scale:(w_offset + self.in_size[1]) * self.scale])
        lr = np.array(lr_crop)
        hr = np.array(hr_crop)

        if random.randint(0, 1):
            lr = lr[:, :, ::-1]
            hr = hr[:, :, ::-1]
        if random.randint(0, 1):
            lr = lr[:, :, :, ::-1]
            hr = hr[:, :, :, ::-1]
        if random.randint(0, 1):
            lr = np.transpose(lr, [0, 1, 3, 2, 4])
            hr = np.transpose(hr, [0, 1, 3, 2, 4])

        self.cur += self.batch_size

        return lr, hr[:, 0]


class TestMinibatch(object):
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            TestMinibatch.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, data_dir, set_file='val.json', batch_size=1, num_frames=5, scale=4):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.scale = scale

        set_file = os.path.join(data_dir, 'sets', set_file)
        with open(set_file, 'r') as fid:
            meta = json.load(fid)

        self.lrcliplist = []
        for vid in meta['videos']:
            if meta['prefix']:
                in_path = os.path.join(data_dir, 'images', meta['x{}_folder'.format(scale)], vid['name'])
            else:
                in_path = os.path.join(data_dir, 'images', vid['name'], meta['x{}_folder'.format(scale)])
            inList = sorted(glob.glob(os.path.join(in_path, '*.png')))

            max_frame = len(inList)
            for i in range(max_frame):
                index = np.array([k for k in range(i - self.num_frames // 2, i + self.num_frames // 2 + 1)])
                index = np.clip(index, 0, max_frame - 1).tolist()
                self.lrcliplist.append([inList[k] for k in index])

        self.total_samples = len(self.lrcliplist)
        self.index = list(range(self.total_samples))
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next()

    def __len__(self):
        return self.total_samples

    def get_next(self):
        if self.cur + self.batch_size > self.total_samples:
            return None

        lrlist = [self.lrcliplist[idx] for idx in self.index[self.cur:self.cur + self.batch_size]]
        lr_names = [self.lrcliplist[idx][self.num_frames//2] for idx in self.index[self.cur:self.cur+self.batch_size]]
        # print(lr_names)
        lr = [np.array([imageio.imread(perimg)[..., :3] / 255. for perimg in pervid]).astype(np.float32) for pervid in lrlist]
        lr = np.array(lr)

        self.cur += self.batch_size

        return lr_names, lr


# =====================================================
# tensorflow interface training dataloader
# =====================================================
def random_flip_lr(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
    return output


def random_flip_ud(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_up_down(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
    return output


def loading_img(output, num_frames):
    target_images = []
    for fi in range(num_frames):
        LR_data = tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(output[fi]), channels=3), dtype=tf.float32)
        target_images.append(LR_data)
    HR_data = tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(output[num_frames]), channels=3), dtype=tf.float32)
    return target_images, HR_data


def preprocess_img(target_images, target_images_hr, num_frames=7, in_size=[64,64], scale=4, lr_shape=[540, 960]):
    flip_decision_lr = tf.random_uniform([], 0, 1, dtype=tf.float32)
    flip_decision_ud = tf.random_uniform([], 0, 1, dtype=tf.float32)
    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(lr_shape[0] - in_size[0], tf.float32) - 1)), dtype=tf.int32)
    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(lr_shape[1] - in_size[1], tf.float32) - 1)), dtype=tf.int32)

    for fi in range(num_frames):
        target_images[fi] = tf.image.crop_to_bounding_box(target_images[fi], offset_h, offset_w, in_size[0], in_size[1])
        target_images[fi] = random_flip_lr(target_images[fi], flip_decision_lr)
        target_images[fi] = random_flip_ud(target_images[fi], flip_decision_ud)

    target_images_hr = tf.image.crop_to_bounding_box(target_images_hr, offset_h*scale, offset_w*scale, in_size[0]*scale, in_size[1]*scale)
    target_images_hr = random_flip_lr(target_images_hr, flip_decision_lr)
    target_images_hr = random_flip_ud(target_images_hr, flip_decision_ud)

    return target_images, target_images_hr


def load_preprocess_tf(output, noise_aug, num_frames=7, in_size=[64,64], scale=4, lr_shape=[540, 960]):
    with tf.name_scope('loading'):
        target_images, target_images_hr = loading_img(output, num_frames)

    with tf.name_scope('data_preprocessing'):
        target_images, target_images_hr = preprocess_img(
            target_images,
            target_images_hr,
            num_frames,
            in_size,
            scale,
            lr_shape)

    target_images = tf.stack(target_images)
    if noise_aug is not None:
        with tf.name_scope('noise_add'):
            target_images = noise_aug.apply_tf(target_images)

    return target_images, target_images_hr


class DataLoader_tensorslice():
    def __init__(self,
                 data_dir,
                 set_file='train.json',
                 batch_size=2,
                 num_frames=7,
                 scale=4,
                 in_size=[32, 32],
                 drop_remainder=True,
                 noise_augmenter=None):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.scale = scale
        self.in_size = in_size
        self.drop_remainder = drop_remainder
        self.noise_augmenter = noise_augmenter

        set_file = os.path.join(data_dir, 'sets', set_file)
        with open(set_file, 'r') as fid:
            meta = json.load(fid)

        self.lrcliplist = []
        self.lrcliplist_img = []

        for vid in meta['videos']:
            if meta['prefix']:
                in_path = os.path.join(data_dir, 'images', meta['x{}_folder'.format(scale)], vid['name'])
                gt_path = os.path.join(data_dir, 'images', meta['gt_folder'], vid['name'])
            else:
                in_path = os.path.join(data_dir, 'images', vid['name'], meta['x{}_folder'.format(scale)])
                gt_path = os.path.join(data_dir, 'images', vid['name'], meta['gt_folder'])
            inList = sorted(glob.glob(os.path.join(in_path, '*.png')))
            gtList = sorted(glob.glob(os.path.join(gt_path, '*.png')))

            vidlen = len(inList)
            assert len(inList) == len(gtList)

            for i in range(0, vidlen-num_frames+1):
                self.lrcliplist.append(inList[i:i+self.num_frames] + [gtList[i+self.num_frames//2]])

            self.lr_shape = vid['x{}_shape'.format(scale)]
            self.hr_shape = vid['gt_shape']

        self.total_samples = len(self.lrcliplist)
        self.index = list(range(self.total_samples))
        random.shuffle(self.index)

        self.build_iterator_namein()

    def build_iterator_namein(self):
        video_dataset = tf.data.Dataset.from_tensor_slices(self.lrcliplist)
        rank_size = int(os.environ['RANK_SIZE'])
        rank_id = int(os.environ['DEVICE_ID'])
        if rank_size > 1:
            print(f'Shard on rank_id {rank_id}')
            video_dataset = video_dataset.shard(rank_size, rank_id)

        video_dataset = video_dataset.shuffle(100000).repeat(300)
        video_dataset = video_dataset.map(
            lambda x: load_preprocess_tf(x, self.noise_augmenter, self.num_frames, self.in_size, self.scale, self.lr_shape),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        video_dataset = video_dataset.batch(self.batch_size, drop_remainder=True)
        video_dataset = video_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        iterator = video_dataset.make_one_shot_iterator()
        self.batch_list = iterator.get_next()


# =====================================================
# tensorflow interface test dataloader
# =====================================================
def loading_test_img(output, num_frames):
    target_images = []
    for fi in range(num_frames):
        LR_data = tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(output[fi]), channels=3), dtype=tf.float32)
        target_images.append(LR_data)
    return output[num_frames//2], tf.stack(target_images)


class DataLoader_tfTest():
    def __init__(self, data_dir, set_file='val.json', batch_size=1, num_frames=7, scale=4):
        self.batch_size = batch_size
        self.num_frames = num_frames

        set_file = os.path.join(data_dir, 'sets', set_file)
        with open(set_file, 'r') as fid:
            meta = json.load(fid)

        self.lrcliplist = []
        for vid in meta['videos']:
            if meta['prefix']:
                in_path = os.path.join(data_dir, 'images', meta['x{}_folder'.format(scale)], vid['name'])
            else:
                in_path = os.path.join(data_dir, 'images', vid['name'], meta['x{}_folder'.format(scale)])
            inList = sorted(glob.glob(os.path.join(in_path, '*.png')))

            max_frames = len(inList)
            for i in range(max_frames):
                index = np.array([k for k in range(i - self.num_frames // 2, i + self.num_frames // 2 + 1)])
                index = np.clip(index, 0, max_frames - 1).tolist()
                self.lrcliplist.append([inList[k] for k in index])

            self.lr_shape = vid['x{}_shape'.format(scale)]

        self.total_samples = len(self.lrcliplist)
        self.index = list(range(self.total_samples))
        self.build_iterator_namein()

    def build_iterator_namein(self):
        video_dataset = tf.data.Dataset.from_tensor_slices(self.lrcliplist)
        rank_size = int(os.environ['RANK_SIZE'])
        rank_id = int(os.environ['DEVICE_ID'])
        if rank_size > 1:
            print(f'Shard on rank_id {rank_id}')
            video_dataset = video_dataset.shard(rank_size, rank_id)

        video_dataset = video_dataset.map(lambda x: loading_test_img(x, self.num_frames),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        video_dataset = video_dataset.batch(self.batch_size, drop_remainder=True)
        video_dataset = video_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        iterator = video_dataset.make_one_shot_iterator()
        self.batch_list = iterator.get_next()

