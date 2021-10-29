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
import sys
import numpy as np
from PIL import Image
import shutil

def preprocessInputData(img):
    img = img.reshape(1,513,513,3)
    return img

if __name__ == "__main__":
    voc_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    seg_trainval_txt = os.path.join(voc_dir,"ImageSets/Segmentation/val.txt")
    with open(seg_trainval_txt,"r") as f:
        lines = f.readlines()
        for line in lines:
            pic_name = line.replace('\n','')
            pic_path = os.path.join(voc_dir,'JPEGImages/{}.jpg'.format(pic_name))
            print("Start to process image {}".format(pic_name))
            origImg = np.array(Image.open(pic_path).resize((513,513)),np.uint8)
            inputValue = preprocessInputData(origImg)
            inputValue.tofile(os.path.join(output_dir,'{}.png.bin'.format(pic_name)))

