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

import os
import PIL
from PIL import Image
import numpy as np




path ="./data/dataset/mlt/"
save_dir="./resized"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir+'/label', exist_ok=True)
os.makedirs(save_dir+'/image', exist_ok=True)

ims = os.path.join(path,"image")
labels = os.path.join(path,"label")

#imgs = os.listdir(ims)[:10]
imgs = os.listdir(ims)
print("num images :", len(imgs))
gts = [elem.replace("jpg","txt") for elem in imgs]
gts = [elem.replace("png","txt") for elem in gts]



for img, gt in zip(imgs, gts):
    im_path = os.path.join(ims,img)
    im = Image.open(im_path).convert("RGB")
    w, h = im.size
    im = im.resize((900,600),resample=PIL.Image.BILINEAR)
    gt_path = os.path.join(labels, gt)
    if not os.path.exists(gt_path):
        print("labels for image {} does not exist".format(gt_path))
        continue

    gt_files = open(gt_path,"r").readlines()
    rects = [f.strip("\n") for f in gt_files]
    print("num_bbox:", len(rects))
    print(gt)
    im.save(save_dir+'/image/'+img)
    new_rect = []
    h_scale = 600.0/h
    w_scale = 900.0/w
    print("w scale:", w_scale)
    print("h scale:", h_scale)
    for rect in rects:
        rect = rect.split(",")
        elem = [int(e) for e in rect]
        elem_rescale = [int(elem[0]*w_scale) ,\
                            int(elem[1]*h_scale) ,\
                            int(elem[2]*w_scale) ,\
                            int(elem[3]*h_scale)]

        rect_str=""
        for elem in elem_rescale:
            rect_str+=str(elem) + ","
        rect_str= rect_str+"\n"
        new_rect.append(rect_str)
    with open(save_dir+"/label/"+gt,'w') as f:
        for line in new_rect:
            f.write(line)


