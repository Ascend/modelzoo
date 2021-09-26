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
import numpy as np
import config as cfg
from augment import DataAugment
import cv2

data_aug = DataAugment()


def polygon_area(poly):
    """
    compute area of a polygon.
    If clockwise order, <0, otherwise >0
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    (h, w) = xxx_todo_changeme
    if len(polys) == 0:
        return polys, tags
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    p_areas = [polygon_area(p) for p in polys]
    to_keep = [idx for idx in range(len(polys)) if abs(p_areas[idx]) > cfg.min_area_not_validate]
    validated_polys = polys[to_keep]
    validated_tags = tags[to_keep]
    return validated_polys, validated_tags


def shrink_poly(poly, shrink_ratio=cfg.shrink_ratio):
    """
    shrink the poly by shrink_ratio, which equals to
    """
    poly = poly.astype(np.float32)
    side_lens = [0.01 + np.linalg.norm(poly[(i + 1) % 4] - poly[i]) for i in range(4)]
    r = [min(side_lens[i], side_lens[(i + 3) % 4]) for i in range(4)]
    dv01, dv03 = (poly[1] - poly[0]) / side_lens[0], (poly[3] - poly[0]) / side_lens[3]
    dv10, dv12 = -dv01, (poly[2] - poly[1]) / side_lens[1]
    dv21, dv23 = -dv12, (poly[3] - poly[2]) / side_lens[2]
    dv30, dv32 = -dv03, -dv23

    shrinked_poly = np.zeros_like(poly)
    shrinked_poly[0] = poly[0] + shrink_ratio * r[0] * (dv01 + dv03)
    shrinked_poly[1] = poly[1] + shrink_ratio * r[1] * (dv10 + dv12)
    shrinked_poly[2] = poly[2] + shrink_ratio * r[2] * (dv21 + dv23)
    shrinked_poly[3] = poly[3] + shrink_ratio * r[3] * (dv30 + dv32)
    return shrinked_poly


def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int) -> tuple:
    # the images are rescaled with ratio (0.5, 1.0, 2.0, 3.0) randomly
    im, text_polys = data_aug.random_scale_with_max_constrain(im, text_polys, scales)
    # the images are horizontally flipped and rotated in range [-10°,10°] randomly and rotated by ±90° randomly
    if np.random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if np.random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    if np.random.random() < 0.1:
        direction = 1 if np.random.random() < 0.5 else 0
        im, text_polys = data_aug.transpose_rotate(im, text_polys, direction)
    return im, text_polys


def order_rotate_rectangle(poly):
    # rbox is clock-wise, with -90°<angle<=0°
    rect = cv2.minAreaRect(poly)
    rbox = np.array(cv2.boxPoints(rect))
    angle = rect[-1]
    angle = angle - 90
    assert (angle <= 0)
    if angle < -45.:
        idx_start = np.argmin(rbox[:, 1])
        angle = -angle - 90.
    else:  # [-45,0)
        idx_start = np.argmin(rbox[:, 0])
        angle = -angle

    rbox = rbox[[idx_start, (idx_start + 1) % 4, (idx_start + 2) % 4, (idx_start + 3) % 4]]
    angle = angle * np.pi / 180.
    return rbox, angle


def point_dist_to_line(vec, dir_vec):
    # assert dir_vec is normalized
    return np.linalg.norm(np.cross(vec, dir_vec))


def generate_label_maps(im_size, text_polys, text_tags):
    """
    Generate training label maps and mask, {1: text, 0: background},
    输出label map尺寸为im_size，这里没有考虑采样
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    training_mask = np.ones((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 4), dtype=np.float32)
    angle_map = np.zeros((h, w), dtype=np.float32)

    for idx, (poly, tag) in enumerate(zip(text_polys.astype(np.int32), text_tags)):
        poly_shrinked = shrink_poly(poly).astype(np.int32)[np.newaxis, :, :]
        cv2.fillConvexPoly(score_map, poly_shrinked, 1)
        cv2.fillConvexPoly(poly_mask, poly_shrinked, idx + 1)
        # if the poly is too small, then ignore it during training.
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < cfg.min_text_size or tag:
            cv2.fillConvexPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        rbox, rotate_angle = order_rotate_rectangle(poly)
        p0_rect, p1_rect, p2_rect, p3_rect = rbox
        dir_p01 = (p1_rect - p0_rect) / np.linalg.norm(p1_rect - p0_rect)
        dir_p12 = (p2_rect - p1_rect) / np.linalg.norm(p2_rect - p1_rect)
        dir_p23 = (p3_rect - p2_rect) / np.linalg.norm(p3_rect - p2_rect)
        dir_p30 = (p0_rect - p3_rect) / np.linalg.norm(p0_rect - p3_rect)

        xy_in_poly = np.argwhere(poly_mask == (idx + 1))
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(point - p0_rect, dir_p01)
            # right
            geo_map[y, x, 1] = point_dist_to_line(point - p1_rect, dir_p12)
            # down
            geo_map[y, x, 2] = point_dist_to_line(point - p2_rect, dir_p23)
            # left
            geo_map[y, x, 3] = point_dist_to_line(point - p3_rect, dir_p30)
            # angle
            angle_map[y, x] = rotate_angle
    label_maps = np.dstack((score_map.astype(np.float32), geo_map, angle_map))
    return label_maps, training_mask


def image_label(im: np.ndarray, text_polys: np.ndarray, text_tags: list, input_size: int, shrink_ratio: float = 0.3,
                degrees: int = 10, scales: np.ndarray = np.array([0.5, 1., 2., 3.])) -> tuple:
    """
    Read Image and Generate Corresponding Label maps.
    在这里必须注意label map生成顺序，必须是先crop后lable，除非交换后对lable结果无影响。
    另外，由于feat map为1/4分辨率，因此也需要考虑先采样后lable，除非交换后对label结果几乎无影响。
    :param im:
    :param text_polys:
    :param text_tags: ignore if True
    :param input_size: net input size, int
    :param shrink_ratio: default 0.3
    :param degrees:
    :param scales:
    :return:
    """
    h, w, _ = im.shape
    # check overflow in text_polys.
    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
    im, text_polys = augmentation(im, text_polys, scales, degrees)
    # random crop care text completeness. output img with size [input_size, input_size, 3]
    im, text_polys, text_tags = data_aug.random_crop_care_text_completeness(im, text_polys, text_tags,
                                                                            (input_size, input_size))
    # generate label maps. ie. EAST: {bbox, angle, score-map}
    h, w, _ = im.shape
    label_maps, training_mask = generate_label_maps((h, w), text_polys, text_tags)

    return im, label_maps, training_mask
