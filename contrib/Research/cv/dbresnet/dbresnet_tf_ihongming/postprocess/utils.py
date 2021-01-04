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
# coding=utf-8
import math

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf

from shapely.geometry import Polygon, MultiPoint

import networks.model as model


def quad_iou(_gt_bbox, _pre_bbox):
    gt_poly = Polygon(_gt_bbox).convex_hull
    pre_poly = Polygon(_pre_bbox).convex_hull

    union_poly = np.concatenate((_gt_bbox, _pre_bbox))

    if not gt_poly.intersects(pre_poly):
        iou = 0
        return iou
    else:
        inter_area = gt_poly.intersection(pre_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area

        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area

        return iou


def polygon_riou(pred_box, gt_box):
    """
    :param pred_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param gt_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
    """
    pred_polygon_points = np.array(pred_box).reshape(-1, 2)
    pred_poly = Polygon(pred_polygon_points).convex_hull

    gt_polygon_points = np.array(gt_box).reshape(-1, 2)

    gt_poly = Polygon(gt_polygon_points).convex_hull
    if not pred_poly.intersects(gt_poly):
        iou = 0
    else:
        inter_area = pred_poly.intersection(gt_poly).area
        union_area = gt_poly.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    return iou


def compute_f1_score(precision, recall):
    if precision == 0 or recall == 0:
        return 0.0
    else:
        return 2.0 * (precision * recall) / (precision + recall)


def load_ctw1500_labels(path):
    """
    load pts
    :param path:
    :return: polys shape [N, 14, 2]
    """

    assert os.path.exists(path), '{} is not exits'.format(path)
    polys = []
    tags = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            x = float(parts[0])
            y = float(parts[1])
            pts = [float(i) for i in parts[4:32]]
            poly = np.array(pts) + [x, y] * 14
            polys.append(poly.reshape([-1, 2]))
            tags.append(False)
    return np.array(polys, np.float), tags


def load_each_image_lable(lable_path):
    lines = []
    reader = open(lable_path, 'r').readlines()
    for line in reader:
        item = {}
        parts = line.strip().split(',')
        label = parts[-1]
        num_points = math.floor((len(parts) - 1) / 2) * 2
        poly = np.array(list(map(float, parts[:num_points]))).reshape((-1, 2)).tolist()
        item['points'] = [tuple(e) for e in poly]
        item['text'] = label
        if label == "###":
            item['ignore'] = True
        else:
            item['ignore'] = False
        height = max(np.array(poly)[:, 1]) - min(np.array(poly)[:, 1])
        width = max(np.array(poly)[:, 0]) - min(np.array(poly)[:, 0])
        if not item['ignore'] and min(height, width) < 8:
            item['ignore'] = True
        lines.append(item)
    return lines


def load_total_text_labels(path):
    assert os.path.exists(path), '{} is not exits'.format(path)
    polys = []
    tags = []
    reader = open(path, 'r').readlines()
    for line in reader:
        parts = line.strip().split(',')
        num_points = math.floor((len(parts) - 1) / 2) * 2
        poly = np.array(list(map(float, parts[:num_points]))).reshape((-1, 2))
        polys.append(poly)
        if parts[-1] == "###":
            tags.append(True)
        else:
            tags.append(False)
        height = max(poly[:, 1]) - min(poly[:, 1])
        width = max(poly[:, 0]) - min(poly[:, 0])
        if parts[-1] != "###" and min(height, width) < 8:
            tags.append(True)
    return np.array(polys), tags


def load_icdar_labels(path):
    pass


def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def resize_img(img, max_size=736):
    h, w, _ = img.shape

    if max(h, w) > max_size:
        ratio = float(max_size) / h if h > w else float(max_size) / w
    else:
        ratio = 1.

    resize_h = int(ratio * h)
    resize_w = int(ratio * w)

    resize_h = resize_h if resize_h % 32 == 0 else abs(resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else abs(resize_w // 32 - 1) * 32
    resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return resized_img, (ratio_h, ratio_w)
