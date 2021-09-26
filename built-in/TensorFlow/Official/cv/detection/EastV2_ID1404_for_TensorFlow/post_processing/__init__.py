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

# -*- coding: utf-8 -*-
# @Time    : 2020/7/16 20:00
# @Author  : yang xuehang
import os
import math
import cv2
import subprocess
import numpy as np
from .pse import pse_cpp, get_num, get_points

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def order_clockwise(poly):
    """
    4 coner vertex, clockwise order, always start in 2nd Quadrant
    """
    cent = np.mean(poly, axis=0)
    poly_cented = poly - cent    
    poly_angles = [math.atan2(p[1], p[0]) for p in poly_cented]
    poly = poly[np.argsort(poly_angles)]
    # compare statpoint
    return poly


def order_rotate_rectangle(poly):
    # Only used when geometry == 'RBOX':
    # rbox is clock-wise, with -90<angle<=0
    # more details of cv2.minAreaRect can be found in notebooks
    rect = cv2.minAreaRect(poly)
    rbox = np.array(cv2.boxPoints(rect))
    angle = rect[-1]
    assert (angle <= 0)
    if angle < -45.:
        idx_start = np.argmin(rbox[:, 1])
        angle = -angle - 90.
    else:  # [-45,0)
        idx_start = np.argmin(rbox[:, 0])
        angle = -angle

    rbox = rbox[[idx_start, (idx_start+1)%4, (idx_start+2)%4, (idx_start+3)%4]]
    angle = angle * np.pi / 180.
    return rbox, angle


def get_side_mask(score_map, label_pts, label_num):
    # head_mask, tail_mask: utilizing info from score_map.
    head_mask = np.zeros_like(score_map, dtype=np.int32)
    tail_mask = np.zeros_like(score_map, dtype=np.int32)
    is_vert = np.array([0]*label_num, dtype=np.int32)
    
    for label_val, label_pt in label_pts.items():
        label_pt = label_pt[2:]
        points = np.array(label_pt, dtype=int).reshape(-1,2)
       
        rbox, angle = order_rotate_rectangle(points)
        p0, p1, p2, p3 = rbox
        w, h = np.linalg.norm(p0-p1), np.linalg.norm(p0-p3)
        # head, tail 
        if h<w:
            dir_v = (p1-p0)*h/w
            head = np.array([p0, p0+dir_v, p3+dir_v, p3], dtype=np.int32)
            tail = np.array([p1-dir_v, p1, p2, p2-dir_v], dtype=np.int32)
            is_vert[label_val] = 0 # horz.
        else:
            dir_v = (p3-p0)*w/h
            head = np.array([p0, p1, p1+dir_v, p0+dir_v], dtype=np.int32)
            tail = np.array([p3-dir_v, p2-dir_v, p2, p3], dtype=np.int32)
            is_vert[label_val] = 1 # vert.
        cv2.fillConvexPoly(head_mask, head, label_val)
        cv2.fillConvexPoly(tail_mask, tail, label_val)
    return head_mask, tail_mask, is_vert 


def decode(score_map, geo_rbox_map, geo_quad_map, score_thresh=0.8, min_area=10):
    """
    distinguish of text and background, give out bbox using geo maps and quad maps.
    :return: quads and confidents 
    """
    kernel = score_map > score_thresh # shrinked text kernel.
    label_num, label_map = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_pts = get_points(label_map, score_map, label_num)

    head_mask, tail_mask, is_vert = get_side_mask(score_map, label_pts, label_num)

    quads_confs = pse_cpp(score_map, label_map, label_num, head_mask, tail_mask, 
                            is_vert, geo_quad_map)
    quads = quads_confs[:,:8]
    confs = quads_confs[:,8:]
    # post-fitlering
    quads = np.array(quads, dtype=np.float)
    keep = []
    for label_idx, label_pt in label_pts.items():
        score_i, count_i = label_pt[0], label_pt[1]
        if count_i < min_area or score_i<0.8:
            continue
        if quads[label_idx][8]<0.8:
            continue
        keep.append(label_idx) 
        quads[label_idx] = order_clockwise(quads[label_idx])
    quads_keep = quads[keep]
    return quads_keep


#def predict(score_map, side_map, geo_map, scale=1, score_thresh=0.5, side_thresh=0.5, min_area=10):
#    from .pse import pse_cpp, get_num, get_points
#    score_map = np.squeeze(score_map)
#    side_map = np.squeeze(side_map)
#    geo_map = np.squeeze(geo_map)
#    #print('score, sides, geos with shape:', score.shape, sides.shape, geos.shape) 
#    kernel = score_map > score_thresh
#    sides = side_map > side_thresh
#    
#    label_num, label_map = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
#    label_values = []
#    label_sum = get_num(label_map, label_num)
#    for label_idx in range(1, label_num):
#        if label_sum[label_idx] < min_area:
#            continue
#        label_values.append(label_idx)
#    quads = pse_cpp(label_map.astype(np.int32), sides.astype(np.uint8), geo_map.astype(np.float32), label_num, scale)
#    quads = np.array(quads, dtype=np.float)
#    for label_idx in range(1, label_num):
#        if any(quads[label_idx][:4]):
#            # TODO: re-guess the right
#            pass
#        if any(quads[label_idx][4:]):
#            # TODO: re-guess the left
#            pass
#    quads = quads.reshape((-1, 4, 2))
#    for label_idx in range(1, label_num):
#        quads[label_idx] = order_clockwise(quads[label_idx])
#    return quads


