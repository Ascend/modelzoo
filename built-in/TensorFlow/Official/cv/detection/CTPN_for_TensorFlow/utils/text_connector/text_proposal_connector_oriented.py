# coding:utf-8
#
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
#

import numpy as np

from utils.text_connector.text_proposal_graph_builder import TextProposalGraphBuilder


class TextProposalConnector:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes
        
        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 棣栧厛杩樻槸寤哄浘锛岃幏鍙栧埌鏂囨湰琛岀敱鍝嚑涓皬妗嗘瀯鎴

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 姣忎釜鏂囨湰琛岀殑鍏ㄩ儴灏忔
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 姹傛瘡涓涓皬妗嗙殑涓績x锛寉鍧愭爣
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)  # 澶氶」寮忔嫙鍚堬紝鏍规嵁涔嬪墠姹傜殑涓績搴楁嫙鍚堜竴鏉＄洿绾匡紙鏈灏忎簩涔橈級

            x0 = np.min(text_line_boxes[:, 0])  # 鏂囨湰琛寈鍧愭爣鏈灏忓
            x1 = np.max(text_line_boxes[:, 2])  # 鏂囨湰琛寈鍧愭爣鏈澶у

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 灏忔瀹藉害鐨勪竴鍗

            # 浠ュ叏閮ㄥ皬妗嗙殑宸︿笂瑙掕繖涓偣鍘绘嫙鍚堜竴鏉＄洿绾匡紝鐒跺悗璁＄畻涓涓嬫枃鏈x鍧愭爣鐨勬瀬宸︽瀬鍙冲搴旂殑y鍧愭爣
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 浠ュ叏閮ㄥ皬妗嗙殑宸︿笅瑙掕繖涓偣鍘绘嫙鍚堜竴鏉＄洿绾匡紝鐒跺悗璁＄畻涓涓嬫枃鏈x鍧愭爣鐨勬瀬宸︽瀬鍙冲搴旂殑y鍧愭爣
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 姹傚叏閮ㄥ皬妗嗗緱鍒嗙殑鍧囧间綔涓烘枃鏈鐨勫潎鍊

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 鏂囨湰琛屼笂绔 绾挎 鐨剏鍧愭爣鐨勫皬鍊
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 鏂囨湰琛屼笅绔 绾挎 鐨剏鍧愭爣鐨勫ぇ鍊
            text_lines[index, 4] = score  # 鏂囨湰琛屽緱鍒
            text_lines[index, 5] = z1[0]  # 鏍规嵁涓績鐐规嫙鍚堢殑鐩寸嚎鐨刱锛宐
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 灏忔骞冲潎楂樺害
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 鏍规嵁楂樺害鍜屾枃鏈涓績绾匡紝姹傚彇鏂囨湰琛屼笂涓嬩袱鏉＄嚎鐨刡鍊
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 宸︿笂
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 鍙充笂
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 宸︿笅
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 鍙充笅
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 鏂囨湰琛屽搴

            fTmp0 = y3 - y1  # 鏂囨湰琛岄珮搴
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 鍋氳ˉ鍋
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x4
            text_recs[index, 5] = y4
            text_recs[index, 6] = x3
            text_recs[index, 7] = y3
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
