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
import math
from utils.bbox_helper import center2corner, corner2center


class Anchors:
    def __init__(self, cfg):
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3] # 宽高比
        self.scales = [8]   # box尺寸
        self.round_dight = 0   # 截断位数
        self.image_center = 0   # 基础锚点的中心在原点
        self.size = 0
        self.anchor_density = 1   # anchor的密度，即每隔几个像素产生锚点

        self.anchor_num = len(self.scales) * len(self.ratios) * (self.anchor_density ** 2)
        self.anchors = None  # 某一像素点的anchor,维度为（anchor_num*4)
        self.all_anchors = None  # in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        """
        生成anchors；1. 用检测区域的长度除以步长得到生成anchor的点；
                    2. 计算生成anchor的点相对于原点的偏移
                    3. 利用meshgrid生成x，y方向的偏移值，确定锚点
                    4. 遍历锚点，生成anchors
        :return: update self.anchors
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        size = self.stride * self.stride
        count = 0
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density) * anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:
                if self.round_dight > 0:
                    ws = round(math.sqrt(size * 1. / r), self.round_dight)
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size * 1. / r))
                    hs = int(ws * r)

                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [-w * 0.5 + x_offset, -h * 0.5 + y_offset,\
                                              w * 0.5 + x_offset, h * 0.5 + y_offset][:]
                    count += 1

    def generate_all_anchors(self, im_c, size):
        """
        生成整幅图像的anchors
        :param im_c: 中心点
        :param size: 图像的size
        :return: 更新 self.all_anchors
        """
        if self.image_center == im_c and self.size == size:
            return False

        self.image_center = im_c
        self.size = size

        anchor0_x = im_c - size // 2 * self.stride
        ori = np.array([anchor0_x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori   # 以图像中心点为中心点的anchor

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        # disp_x是[1, 1, size]，disp_y是[1, size, 1]
        disp_x = np.arange(0, size).reshape(1, 1, size) * self.stride
        disp_y = np.arange(0, size).reshape(1, size, 1) * self.stride

        # 得到整幅图像中anchor中心点的坐标
        cx = cx + disp_x
        cy = cy + disp_y

        # 通过广播生成整幅图像的anchor broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])
        # 以中心点坐标,宽高和左上角、右下角坐标两种方式存储anchors
        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])

        return True

if __name__ == '__main__':
    anchors = Anchors(cfg={'stride':16, 'anchor_density': 2})
    anchors.generate_all_anchors(im_c=255//2, size=(255-127)//16+1+8)

