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
import cv2
import numpy as np
import math
import numbers
from skimage.util import random_noise
import config as cfg


def show_pic(img, bboxes=None, name='pic'):
    show_img = img.copy()
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    for point in bboxes.astype(np.int):
        cv2.line(show_img, tuple(point[0]), tuple(point[1]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[1]), tuple(point[2]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[2]), tuple(point[3]), (255, 0, 0), 2)
        cv2.line(show_img, tuple(point[3]), tuple(point[0]), (255, 0, 0), 2)
    cv2.imshow(name, show_img)


class DataAugment(object):
    def __init__(self):
        pass

    @staticmethod
    def add_noise(im: np.ndarray):
        """ Add gaussian white noise. """
        return (random_noise(im, mode='gaussian', clip=True) * 255).astype(im.dtype)

    @staticmethod
    def random_scale_with_max_constrain(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray or list,
                                        max_len_size: int = 2048) -> tuple:
        """
        Random choose a scale in scale list, then resize im and text_polys. Note the max edge should be contained by max_len_size.
        """
        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(scales))
        h, w, _ = im.shape
        if max(h, w) * rd_scale > max_len_size:
            rd_scale = float(max_len_size) / max(h, w)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale
        return im, tmp_text_polys

    @staticmethod
    def random_scale(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray or list) -> tuple:
        """
        Random choose a scale in scale list, then resize im and text_polys
        :param im:
        :param text_polys:
        :param scales:
        :return:
        """
        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale
        return im, tmp_text_polys

    @staticmethod
    def random_rotate_img_bbox(img, text_polys: np.ndarray,
                               degrees: numbers.Number or list or tuple or np.ndarray, same_size=False):
        """
        Random choose an angle, then rotate im and text_polys
        :param img:
        :param text_polys:
        :param degrees:
        :param same_size:
        :return:
        """

        def parse_degrees(degrees: numbers.Number or list or tuple or np.ndarray):
            if isinstance(degrees, numbers.Number):
                if degrees < 0.:
                    raise ValueError('If degree is a single number, it must be postive.')
                return -degrees, degrees
            elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
                if len(degrees) != 2:
                    raise ValueError('If degree is a sequence, it must be of len 2.')
                return degrees
            else:
                raise Exception('degrees must in Number or list or tuple or np.ndarray')

        degrees = parse_degrees(degrees)
        # ---- rotate img -------#
        h, w = img.shape[0:2]
        angle = np.random.uniform(degrees[0], degrees[1])
        if same_size:
            nh, nw = h, w
        else:
            rangle = np.deg2rad(angle)
            nw = abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)
            nh = abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)
        # rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # affine rotate
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        # affine rotate of polygons
        rot_text_polys = np.append(text_polys, np.ones([len(text_polys), 4, 1]), axis=-1)
        rot_text_polys = np.dot(rot_text_polys, rot_mat.T)
        return rot_img, np.array(rot_text_polys, dtype=np.float32)

    @staticmethod
    def horizontal_flip(im: np.ndarray, text_polys: np.ndarray) -> tuple:
        # horizontal flip
        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]
        return flip_im, flip_text_polys

    @staticmethod
    def vertical_flip(im: np.ndarray, text_polys: np.ndarray) -> tuple:
        # vertical flip
        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        return flip_im, flip_text_polys

    @staticmethod
    def transpose_rotate(img: np.ndarray, text_polys: np.ndarray, direction=0):
        """
        Random rotate img with 90°(0) or -90°(1), anti-clockwise.
        """
        assert (direction in [0, 1])
        rot_img = cv2.transpose(img)
        rot_text_polys = text_polys[:, :, ::-1]
        rot_img = cv2.flip(rot_img, direction)
        h, w, _ = rot_img.shape
        if direction == 0:  # vert flip
            rot_text_polys[:, :, 1] = h - rot_text_polys[:, :, 1]  # ←↑
        else:  # horz flip
            rot_text_polys[:, :, 0] = w - rot_text_polys[:, :, 0]  # →↓
        return rot_img, rot_text_polys

    @staticmethod
    def random_crop_care_text_completeness(img, polys: np.ndarray, tags: np.ndarray, img_size: tuple,
                                           hard_mode=True):
        """
        Random crop training images from img, note that text completeness should be noticed.
            Vertical text: can only be cut horizontally, in midden part, nor in either side.
            Horizontal text: can only be cut vertically, in midden part, nor in either side.
        :param img:
        :param polys:
        :param tags:
        :param img_size: target size (h,w).
        :param hard_mode: hard mode, text cannot be cutted if True. otherwise, text can be cut in midden parts.
        :return:
        """
        def get_crop_ok_array(img, polys):
            h, w = img.shape[0:2]
            pad_h, pad_w = h // 10, w // 10
            h_array = np.zeros((h + 2 * pad_h), dtype=np.int32)
            w_array = np.zeros((w + 2 * pad_w), dtype=np.int32)
            tmp_polys = np.round(polys.copy(), decimals=0).astype(np.int32)
            xs_min, ys_min = np.min(tmp_polys[:, :, 0], axis=1), np.min(tmp_polys[:, :, 1], axis=1)
            xs_max, ys_max = np.max(tmp_polys[:, :, 0], axis=1), np.max(tmp_polys[:, :, 1], axis=1)
            for xmin, ymin, xmax, ymax in zip(xs_min, ys_min, xs_max, ys_max):
                w_array[xmin + pad_w:xmax + pad_w] = 1
                h_array[ymin + pad_h:ymax + pad_h] = 1
                if not hard_mode:  # soft mode.
                    if ymax - ymin > 3 * (xmax - xmin):  # i.e. vertical text
                        h_array[ymin + pad_h + (xmax - xmin):ymax + pad_h - (xmax - xmin)] = 0
                    elif xmax - xmin > 3 * (ymax - ymin):  # i.e. horizontal text
                        w_array[xmin + pad_w + (ymax - ymin):xmax + pad_w - (ymax - ymin)] = 0
            return w_array, h_array, pad_h, pad_w

        def cropper(img, polys, tags, img_size):
            th, tw = img_size
            h, w, _ = img.shape
            if h == th and w == tw:
                return img, polys, tags
            # padding and crop
            max_h_th, max_w_tw = np.max([h, w, th]), np.max([h, w, tw])
            im_padded = np.zeros((max_h_th, max_w_tw, 3), dtype=np.uint8)
            im_padded[:h, :w, :] = img.copy()
            img = cv2.resize(im_padded, dsize=img_size)
            rd_scale = (tw / float(max_w_tw), th / float(max_h_th))
            if len(polys) > 0:
                polys[:, :, 0] *= rd_scale[0]
                polys[:, :, 1] *= rd_scale[1]
            return img, polys, tags

        th, tw = img_size
        h, w, _ = img.shape
        # if no polys.
        if len(polys) == 0:
            return cropper(img, polys, tags, img_size)

        # if with polys. get choosable x,y
        w_array, h_array, pad_h, pad_w = get_crop_ok_array(img, polys)
        w_choose = np.where(h_array == 0)[0]
        h_choose = np.where(w_array == 0)[0]
        if len(w_choose) < 2 or len(h_choose) < 2:
            return cropper(img, polys, tags, img_size)

        for i in range(cfg.max_crop_tries):
            xx = np.random.choice(w_choose, size=2) - pad_w
            yy = np.random.choice(h_choose, size=2) - pad_h
            xx = np.clip(xx, 0, w - 1)
            yy = np.clip(yy, 0, h - 1)
            xmin, xmax, ymin, ymax = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
            if xmax - xmin < cfg.min_crop_side_ratio * tw or ymax - ymin < cfg.min_crop_side_ratio * th:
                continue
            index_in_crop = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) & (polys[:, :, 1] >= ymin) & (
                        polys[:, :, 1] <= ymax)
            index_selected = np.where(np.sum(index_in_crop, axis=1) == 4)[0]
            img_crop, polys_crop, tags_crop = img[ymin:ymax+1, xmin:xmax+1, :], polys[index_selected], tags[index_selected]
            if len(index_selected) > 0:
                polys_crop[:, :, 0] -= xmin
                polys_crop[:, :, 1] -= ymin
            elif np.random.random() < 3. / 8.:
                continue
            return cropper(img_crop, polys_crop, tags_crop, img_size)

        return cropper(img, polys, tags, img_size)

    @staticmethod
    def random_crop(img, label_maps, training_mask, img_size):
        """
        Random crop training images from img. output img with size of img_size
        :param img: (h,w,3), np.uint8
        :param label_maps: [score_map, hei_map, angle_map]
        :param training_mask:
        :param img_size:
        :return:
        """
        h, w = img.shape[0:2]
        th, tw = img_size
        if h == th and w == tw:
            return img, label_maps, training_mask
        # make sure text exists in label map, and crop image by given prob.
        if np.max(label_maps[:, :, 0]) and np.random.random() > 7. / 8.:
            # top left text instances.+
            tl = np.min(np.where(label_maps[:, :, 0] > 0), axis=1) - img_size
            tl[tl < 0] = 0
            # bottom right text instances.
            br = np.max(np.where(label_maps[:, :, 0] > 0), axis=1) - img_size
            br[br < 0] = 0
            # make sure the rl can be covered.
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)
            # make sure crop img have text.
            for _ in range(50000):
                i = np.random.randint(tl[0], br[0] + 1)
                j = np.random.randint(tl[1], br[1] + 1)
                if label_maps[:, :, 0][i:i + th, j:j + tw].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = np.random.randint(0, h - th + 1)
            j = np.random.randint(0, w - tw + 1)
        # return i,j,th,tw
        img = img[i:i + th, j:j + tw]
        label_maps = label_maps[i:i + th, j:j + tw, :]
        training_mask = training_mask[i:i + th, j:j + tw]
        return img, label_maps, training_mask

    def test(self, im: np.ndarray, text_polys: np.ndarray):
        print('Random scale')
        t_im, t_text_polys = self.random_scale(im, text_polys, [0.5, 1, 2, 3])
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_scale')

        print('Random rotate')
        t_im, t_text_polys = self.random_rotate_img_bbox(im, text_polys, 10)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'random_rotate_img_bbox')

        print('Horizontal flip')
        t_im, t_text_polys = self.horizontal_flip(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'horizontal_flip')

        print('Vertical flip')
        t_im, t_text_polys = self.vertical_flip(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'vertical_flip')
        show_pic(im, text_polys, 'vertical_flip_ori')

        print('horizontal to vertical transpose')
        t_im, t_text_polys = self.transpose_rotate(im, text_polys)
        print(t_im.shape, t_text_polys.dtype)
        show_pic(t_im, t_text_polys, 'transpose_rotate')
        show_pic(im, text_polys, 'transpose_rotate_ori')

        print('Adding noise')
        t_im = self.add_noise(im)
        print(t_im.shape)
        show_pic(t_im, text_polys, 'add_noise')
        show_pic(im, text_polys, 'add_noise_ori')
