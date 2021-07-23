#!/usr/bin/python3
# coding=utf-8
# Copied from
# https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess/
# In-depth article about the algorithm:
# https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
# Refactored to reduce complexity
# changed to process peak coordinates from tensorflow::ops::Where
# instead of peaks map from python tf.Where
# ======================================================================
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

import logging
import math
import os
import sys
from enum import Enum
from logging import handlers

import cv2
import numpy as np
import scipy.stats as st
import tensorflow.compat.v1 as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.FATAL)

try:
    from pafprocess import pafprocess
except ModuleNotFoundError:
    print('Need to build c++ library for pafprocess.'
          'SEE pafprocess path README.md')
    sys.exit(-1)

COCO_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0],
    [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170],
    [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

COCO_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6),
    (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17), (2, 16), (5, 17)
]  # = 19

COCO_PAIRS_RENDER = COCO_PAIRS[:-2]


def read_img_file(file, width=None, height=None):
    val_image = cv2.imread(file, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def round_int(val):
    return int(round(val))


def write_coco_json(human, image_w, image_h):
    key_points = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            key_points.extend([0, 0, 0])
            continue
        body_part = human.body_parts[coco_id]
        key_points.extend([round_int(body_part.x * image_w), round_int(body_part.y * image_h), 2])
    return key_points


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)


def _include_part(part_list, part_idx):
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Logging:
    def __init__(self):
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='%(asctime)s - %(levelname[0])s - [%(name)s]: %(message)s'
        # )
        filename = 'predict_openpose.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s]: %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        th = handlers.TimedRotatingFileHandler(filename=filename, when='D', backupCount=10, encoding='utf-8')
        th.setFormatter(formatter)
        self.logger = logging.getLogger('TfPose')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

    def info(self, info=""):
        self.logger.info(info)

    def error(self, error=""):
        self.logger.error(error)

    def warn(self, warning=""):
        self.logger.warning(warning)

    def critical(self, critical=""):
        self.logger.critical(critical)


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _round(v):
        return int(round(v))

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)

        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:
            # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - self._round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if self._round(x2 - x) == 0.0 or self._round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": self._round((x + x2) / 2),
                    "y": self._round((y + y2) / 2),
                    "w": self._round(x2 - x),
                    "h": self._round(y2 - y)}
        else:
            return {"x": self._round(x),
                    "y": self._round(y),
                    "w": self._round(x2 - x),
                    "h": self._round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8

        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -
        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if self._round(x2 - x) == 0.0 or self._round(y2 - y) == 0.0:
            return None
        return {"x": self._round((x + x2) / 2),
                "y": self._round((y + y2) / 2),
                "w": self._round(x2 - x),
                "h": self._round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:

    def __init__(self, target_size=(320, 240)):
        self.target_size = target_size
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')
        self.out = tf.placeholder(dtype=tf.float32, shape=(1, 46, 82, 57), name='output')
        self.tensor_heatMat = self.out[:, :, :, :19]
        self.tensor_pafMat = self.out[:, :, :, 19:]
        self.tensor_heatMat_up = tf.image.resize_area(self.out[:, :, :, :19], self.upsample_size,
                                                      align_corners=False, name='upsample_heatmat')
        self.tensor_pafMat_up = tf.image.resize_area(self.out[:, :, :, 19:], self.upsample_size,
                                                     align_corners=False, name='upsample_pafmat')

        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heat_mat = smoother.get_output()

        max_pooled_in_tensor = tf.nn.pool(gaussian_heat_mat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heat_mat, max_pooled_in_tensor), gaussian_heat_mat,
                                     tf.zeros_like(gaussian_heat_mat))

        self.heatMat = self.pafMat = None

    def __del__(self):
        # self.persistent_sess.close()
        pass

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped

    def inference(self, file_name, resize_to_default=True, upsample_size=1.0):

        npimg = np.fromfile(file_name, "float32").reshape(1, 46, 82, 57)

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            peaks, heat_mat_up, paf_mat_up = sess.run(
                [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up], feed_dict={
                    self.out: npimg, self.upsample_size: upsample_size})

        peaks = peaks[0]
        self.heatMat = heat_mat_up[0]
        self.pafMat = paf_mat_up[0]
        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat)
        return humans


class Smoother(object):
    def __init__(self, inputs, filter_size, sigma, heat_map_size=0):
        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.filter_size = filter_size
        self.sigma = sigma
        self.heat_map_size = heat_map_size
        self.setup()

    def setup(self):
        self.feed('data').conv(name='smoothing')

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    @staticmethod
    def gauss_kernel(kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / kernlen
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def make_gauss_var(self, name, size, sigma, c_i):
        kernel = self.gauss_kernel(size, sigma, c_i)
        var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
        return var

    def get_output(self):
        """Returns the smoother output."""
        return self.terminals[-1]

    @layer
    def conv(self, feature_map, name, padding='SAME'):
        # Get the number of channels in the input
        if self.heat_map_size != 0:
            c_i = self.heat_map_size
        else:
            c_i = feature_map.get_shape().as_list()[3]
        # Convolution for a given input and kernel
        convolution = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding=padding)
        with tf.variable_scope(name):
            kernel = self.make_gauss_var('gauss_weight', self.filter_size, self.sigma, c_i)
            output = convolution(feature_map, kernel)
        return output
