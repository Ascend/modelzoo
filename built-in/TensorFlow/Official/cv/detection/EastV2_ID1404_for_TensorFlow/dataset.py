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

#!/usr/bin/env python
# -*- coding:utf-8 -*-
from npu_bridge.npu_init import *
import os
import numpy as np
import cv2
import pathlib
import time

import config as cfg
from data_utils_mt import GeneratorEnqueuer
from data_utils import image_label


class ImageDataset(object):
    def __init__(self, data_list, input_size):
        self.data_list = self.load_data(data_list)
        self.input_size = input_size

    @staticmethod
    def _order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        im = cv2.imread(img_path, 1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img, label_maps, training_mask = image_label(im, text_polys, text_tags, cfg.input_size, cfg.shrink_ratio)
        return img, label_maps, training_mask

    def load_data(self, data_list: list) -> list:
        t_data_list = []
        for img_path, label_path in data_list:
            bboxes, text_tags = self._load_annoataion(label_path)
            if len(bboxes) > 0:
                t_data_list.append((img_path, bboxes, text_tags))
            else:
                print('there is no suit bbox in {}.'.format(label_path))
        return t_data_list

    def _load_annoataion(self, txtname: str) -> tuple:
        text_polys = []
        text_tags = []
        if not os.path.exists(txtname):
            print('File {} not found.'.format(txtname))
            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)
        with open(txtname, 'r', encoding='utf-8') as f:
            for line in f:
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                if not line and len(line) < 8:
                    print('Error line in file {} with {}.'.format(txtname, line))
                    continue
                box = np.array(list(map(float, params[:8]))).reshape((-1,2))
                box = self._order_points_clockwise(box)
                label = ','.join(line[8:])
                text_polys.append(box)
                text_tags.append(True if label == '*' or label == '###' else False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)


def get_datalist(train_data_path):
    """
    Get training and validation dataset.
    :param train_data_path:
    :return:
    """
    train_data_list = []
    for train_path in train_data_path:
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
        train_data_list.extend(train_data)
    return train_data_list


def generator(input_size=512, batch_size=16, image_list=None, vis=False):
    #image_list = np.array(get_datalist(train_data_path))
    #print('{} training images in {}'.format(len(image_list), train_data_path))
    index = np.arange(0, len(image_list))
    dataset = ImageDataset(image_list, input_size)
    while True:
        np.random.shuffle(index)
        images, image_labels, training_masks = [], [], []
        for idx in index:
            img, label_maps, train_mask = dataset[idx]
            if vis:
                visualize_label_maps(img, label_maps)
            # print('one sample has done!')

            images.append(img.astype(np.float32))
            image_labels.append(label_maps[::4, ::4, :])
            training_masks.append(train_mask[::4, ::4, np.newaxis].astype(np.float32))
            # print(training_masks[0].shape, len(training_masks))
            if batch_size == len(images):
                yield images, image_labels, training_masks
                images, image_labels, training_masks = [], [], []


def get_batch(num_workers, **kwargs):
    enqueuer = None
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    # print('sleeping')
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()




def visualize_label_maps(im, label_maps):
    import matplotlib.pyplot as plt
    score_map = label_maps[:, :, 0]
    hei_map = label_maps[:, :, 1]
    ang_map = label_maps[:, :, 2]
    xy_cent_line = np.argwhere(score_map > 0)

    for y, x in xy_cent_line:
        im[y, x, :] = np.array([0, 0, 255], dtype=np.uint8)

    im_slide_bbox = im.copy()
    _cnt = 0
    for y, x in xy_cent_line:
        h, ang = hei_map[y, x], ang_map[y, x]
        rect = ((x, y), (h, h), ang)
        pts = cv2.boxPoints(rect)
        if _cnt % 50 == 0:
            im_slide_bbox = cv2.polylines(im_slide_bbox, [pts.astype(np.int32)], True, (255,0,0), 2)
        _cnt += 1

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(im)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(im_slide_bbox) #label_map
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()
    plt.close()



if __name__ == '__main__':
    gen = generator(vis=True, batch_size=2)
    i = 0
    while i < 1000:
        data = next(gen)
        # data = gen.next()
        i += 1
        # print(data)
        print('len(data):%d' % len(data))

