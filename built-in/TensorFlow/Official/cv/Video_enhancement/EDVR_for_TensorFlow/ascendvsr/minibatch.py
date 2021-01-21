import os
import glob
import imageio
import numpy as np
import random
import json
from multiprocessing import Process, Queue

import tensorflow as tf


class Minibatch(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            Minibatch.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, data_dir, set_file='train.json', batch_size=2, num_frames=7, scale=4, in_size=[32, 32], drop_remainder=True):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.scale = scale
        self.in_size = in_size
        self.drop_remainder = drop_remainder

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


def get_single_example(reader, filename_queue):
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'lr': tf.FixedLenFeature([], tf.string),
            'hr': tf.FixedLenFeature([], tf.string),
        }
    )

    return dict(lr=tf.decode_raw(features['lr'], tf.uint8),
                hr=tf.decode_raw(features['hr'], tf.uint8))


def next_batch(record_dir, batch_size, num_frames, scale, raw_size, in_size, use_shuffle=False):
    reader = tf.TFRecordReader()
    pattern = '{}*'.format(record_dir)
    tfrecord_list = tf.train.match_filenames_once(pattern)
    filename_queue = tf.train.string_input_producer(tfrecord_list)

    example_dict = get_single_example(reader, filename_queue)
    lr = example_dict['lr']
    hr = example_dict['hr']

    lr = tf.reshape(lr, [num_frames, raw_size[0], raw_size[1], 3])
    hr = tf.reshape(hr, [raw_size[0] * scale, raw_size[1] * scale, 3])

    shape = tf.shape(lr)[1:]
    size = tf.convert_to_tensor([in_size[0], in_size[1], 3], dtype=tf.int32, name="size")

    limit = shape - size + 1
    offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

    offset_in = tf.concat([[0], offset], axis=-1)
    size_in = tf.concat([[num_frames], size], axis=-1)
    lr = tf.slice(lr, offset_in, size_in)
    offset_gt = tf.concat([offset[:2] * scale, [0]], axis=-1)
    size_gt = tf.concat([size[:2] * scale, [3]], axis=-1)
    hr = tf.slice(hr, offset_gt, size_gt)

    lr.set_shape([num_frames, in_size[0], in_size[1], 3])
    hr.set_shape([in_size[0] * scale, in_size[1] * scale, 3])

    in_name_list = ['lr', 'hr']
    in_list = [lr, hr]

    if use_shuffle:
        out_list = tf.train.shuffle_batch(in_list, batch_size=batch_size, capacity=100, num_threads=batch_size,
                                          min_after_dequeue=50)
    else:
        out_list = tf.train.batch(in_list, batch_size=batch_size, capacity=batch_size, num_threads=batch_size,
                                  dynamic_pad=True)

    out_dict = dict()
    for idx, name in enumerate(in_name_list):
        out_dict[name] = out_list[idx]

    lr = tf.to_float(out_dict['lr'] / 255)
    hr = tf.to_float(out_dict['hr'] / 255)

    return lr, hr
