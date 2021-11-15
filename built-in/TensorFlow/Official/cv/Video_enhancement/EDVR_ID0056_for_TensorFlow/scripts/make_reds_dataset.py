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
import os
import json
import sys


def gen_set_meta(meta):
    return dict(
        prefix=False,
        gt_folder='truth',
        x4_folder='blur4',
        videos=meta)


def split_sets(set_path, meta, meta4):
    """
    train   0~239
    val     240~269         0 11 15 20
    """
    train_meta = []
    val_meta = []
    for m, m4 in zip(meta, meta4):
        assert m['idx'] == m4['idx']
        assert m['H'] == m4['H'] * 4
        assert m['W'] == m4['W'] * 4
        assert m['nframes'] == m4['nframes']
        k = int(m['idx'])
        o = dict(name=m['idx'],
                 gt_shape=(m['H'], m['W']),
                 x4_shape=(m4['H'], m4['W']),
                 nframes=m['nframes'])
        if k in [0, 11, 15, 20]:
            val_meta.append(o)
        elif 0 <= k <= 269:
            train_meta.append(o)
        else:
            raise KeyError

    if not os.path.exists(set_path):
        os.mkdir(set_path)

    with open(os.path.join(set_path, 'train.json'), 'w') as fid:
        json.dump(gen_set_meta(train_meta), fid)
    with open(os.path.join(set_path, 'val.json'), 'w') as fid:
        json.dump(gen_set_meta(val_meta), fid)


def decode_videos():
    meta = []
    meta4 = []
    for i in range(269):
        meta.append(dict(idx='{:03d}'.format(i), nframes=100, H=720, W=1280))
        meta4.append(dict(idx='{:03d}'.format(i), nframes=100, H=180, W=320))

    return meta, meta4


if __name__ == '__main__':
    #image_path = 'data/reds/images'
    datadir = sys.argv[1]
    set_path = os.path.join(datadir, 'sets')
    meta, meta4 = decode_videos()
    split_sets(set_path, meta, meta4)
