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
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
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

import numpy as np
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N（左下 右上两个点）
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def cxy_wh_2_rect(pos, sz):
    """
    转换矩形框的表示方式
    :param pos: 矩形框中心点坐标
    :param sz: 矩形框大小：宽高
    :return: 矩形框的左上角坐标，宽，高
    """
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def get_axis_aligned_bbox(region):
    """
    将目标区域其最小外接矩形的形式：中心点坐标和宽，高的形式
    :param region:
    :return:中心点坐标，宽，高
    """
    nv = region.size
    # 若region是四角坐标，可能不平行于坐标轴
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        # 计算外接矩形的左上角坐标和右下角坐标
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        # 求L2范数  平行四边形面积
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        # 求宽和高
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    # region 是左上角坐标和宽高
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        center = corner2center(bbox)
        original_center = center

        real_param = {}
        if 'scale' in param:
            # 矩阵缩放
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)

            # center.w *= scale_x
            # center.h *= scale_y
            center = Center(center.x, center.y, center.w * scale_x, center.h * scale_y)

        # 获取目标框（x1, y1, x2, y2 ）
        bbox = center2corner(center)

        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)

        real_param['scale'] = current_center.w / original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - original_center.x, current_center.y - original_center.y

        return bbox, real_param
    else:
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.

        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0

        center = corner2center(bbox)

        center = Center(center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y)

        return center2corner(center)


def IoU(rect1, rect2):
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)

    target_a = (tx2-tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap