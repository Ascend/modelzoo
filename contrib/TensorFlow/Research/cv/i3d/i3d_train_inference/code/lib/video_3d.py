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
import random
import os
import numpy as np
# To suppress the complaint of `image file is truncated`
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image

from lib.data_augment import transform_data


class Video_3D:
    def __init__(self, info_list, tag='rgb', img_format='frame{:06d}{}.jpg'):
        '''
            info_list: [name, path, total_frame, label]
            tag: 'rgb'(default) or 'flow'
            img_format: 'frame{:06d}{}.jpg'(default)
        '''
        #initialzie,to ensure the int is int
        if isinstance(info_list[0], str):
            self.name = info_list[0]
        else:
            self.name = info_list[0].decode('utf-8')
        if isinstance(info_list[1], str):
            self.path = info_list[1]
        else:
            self.path = info_list[1].decode('utf-8')
        # print('self.name:', self.name)
        # print('self.path:', self.path)
        if isinstance(info_list[2], int):
            self.total_frame_num = info_list[2]
        else:
            self.total_frame_num = int(info_list[2])
        if isinstance(info_list[3], int):
            self.label = info_list[3]
        else:
            self.label = int(info_list[3])
        if isinstance(tag, str):
            self.tag = tag
        else:
            self.tag = tag.decode('utf-8')

        #img_format offer the standard name of pic
        self.img_format = img_format

    def get_frames(self, frame_num, side_length=224, is_numpy=True, data_augment=False):
        #assert frame_num <= self.total_frame_num
        frames = list()
        start = random.randint(1, max(self.total_frame_num-frame_num, 0)+1)
        #combine all frames
        for i in range(start, start+frame_num):
            frames.extend(self.load_img((i-1) % self.total_frame_num+1))
        frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        
#?? what is the meaning of is_numpy
        if is_numpy:
            frames_np = []
            if self.tag == 'rgb':
                for i, img in enumerate(frames):
                    frames_np.append(np.asarray(img))
            elif self.tag == 'flow':
                for i in range(0, len(frames), 2):
                    #it is used to combine frame into 2 channels
                    tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
                    frames_np.append(tmp)
            return np.stack(frames_np)

        return frames


    def load_img(self, index):
        img_dir = self.path
        # print('img_dir:', img_dir, self.tag)
        if self.tag == 'rgb':
            img = Image.open(os.path.join(img_dir, self.img_format.format(index, ''))).convert('RGB')
            return [img]
        if self.tag == 'flow':
            # u_img = Image.open(os.path.join(img_dir, self.img_format.format(index, '_u'))).convert('L')
            # v_img = Image.open(os.path.join(img_dir, self.img_format.format(index, '_v'))).convert('L')
            u_img = Image.open(os.path.join(img_dir.format('u'), self.img_format.format(index, ''))).convert('L')
            v_img = Image.open(os.path.join(img_dir.format('v'), self.img_format.format(index, ''))).convert('L')
            return [u_img,v_img]
        return

    def __str__(self):
        return 'Video_3D:\nname: {:s}\nframes: {:d}\nlabel: {:d}\nPath: {:s}'.format(
            self.name, self.total_frame_num, self.label, self.path)
