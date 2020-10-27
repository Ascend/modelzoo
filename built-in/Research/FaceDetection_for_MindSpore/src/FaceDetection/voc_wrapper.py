# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Face detection compute final result."""
import numpy as np
import os

from .nms_wrapper import nms


def remove_5050_face(dst_txt, img_size):
    dst_txt_rm5050 = dst_txt.replace('.txt', '') + '_rm5050.txt'
    if os.path.exists(dst_txt_rm5050):
        os.remove(dst_txt_rm5050)

    write_lines = []
    with open(dst_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            info = line.replace('\n', '').split(' ')
            img_name = info[0]
            size = img_size[img_name][0]
            w = float(info[4]) - float(info[2])
            h = float(info[5]) - float(info[3])
            radio = max(float(size[0]) / 1920., float(size[1]) / 1080.)
            new_w = float(w) / radio
            new_h = float(h) / radio
            if min(new_w, new_h) >= 50.:
                write_lines.append(line)

    file.close()

    with open(dst_txt_rm5050, 'a') as fw:
        for line in write_lines:
            fw.write(line)


def gen_results(reorg_dets, results_folder, img_size, nms_thresh=0.45):
    for label, pieces in reorg_dets.items():
        ret = []
        dst_fp = '%s/comp4_det_test_%s.txt' % (results_folder, label)
        for name in pieces.keys():
            pred = np.array(pieces[name], dtype=np.float32)
            keep = nms(pred, nms_thresh)
            for ik in keep:
                line = '%s %f %s' % (name, pred[ik][-1], ' '.join([str(num) for num in pred[ik][:4]]))
                ret.append(line)

        with open(dst_fp, 'w') as fd:
            fd.write('\n'.join(ret))

        remove_5050_face(dst_fp, img_size)


def reorg_detection(dets, netw, neth, img_sizes):
    reorg_dets = {}
    for k, v in dets.items():
        name = k
        orig_width, orig_height = img_sizes[k][0]
        scale = min(float(netw)/orig_width, float(neth)/orig_height)
        new_width = orig_width * scale
        new_height = orig_height * scale
        pad_w = (netw - new_width) / 2.0
        pad_h = (neth - new_height) / 2.0

        for iv in v:
            xmin = iv.x_top_left
            ymin = iv.y_top_left
            xmax = xmin + iv.width
            ymax = ymin + iv.height
            conf = iv.confidence
            class_label = iv.class_label

            xmin = max(0, float(xmin - pad_w)/scale)
            xmax = min(orig_width - 1, float(xmax - pad_w)/scale)
            ymin = max(0, float(ymin - pad_h)/scale)
            ymax = min(orig_height - 1, float(ymax - pad_h)/scale)

            reorg_dets.setdefault(class_label, {})
            reorg_dets[class_label].setdefault(name, [])
            piece = (xmin, ymin, xmax, ymax, conf)
            reorg_dets[class_label][name].append(piece)

    return reorg_dets
