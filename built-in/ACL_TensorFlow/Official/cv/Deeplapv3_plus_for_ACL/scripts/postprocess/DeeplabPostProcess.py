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

import numpy as np
import os
import sys
from PIL import Image
from get_dataset_colormap import label_to_color_image

resDir = sys.argv[1]
labelDir = "./labeledSegCls/"
resLis = os.listdir(resDir)
labelLis = os.listdir(labelDir)

def getUnique(img):
    return np.unique(img)

def getIntersection(img, label, i):
    cnter=0
    for h_img, h_label in zip(img, label):
        for w_img, w_label  in zip(h_img, h_label):
            if w_img==i and w_label==i:
                cnter+=1
    return cnter

def getUnion(img, label, i):
    cnter=0
    for h_img, h_label in zip(img, label):
        for w_img, w_label  in zip(h_img, h_label):
            if w_img==i or w_label==i:
                cnter+=1
    return cnter

def getIoU(img, label):
    uniqueVals = getUnique(img)
    iou = 0.0
    cnter = 0
    for i in uniqueVals:
        if i==0:
            continue
        cnter+=1
        intersection = getIntersection(img, label, i)
        union = getUnion(img, label, i)
        iou+=float(intersection)/union
    if cnter==0:
        return 0
    else:
        return iou/cnter
cnter=0
accumulatedIOU=0.0
for r in resLis:
    cnter+=1
    rVal = np.fromfile(resDir+r, np.int32).astype(np.uint8).reshape(375, 500)
    newName = r.split("_")[1]+"_"+r.split("_")[2]
    lVal = np.array(Image.open(labelDir+newName).resize((500,375)),np.uint8)
    iou = getIoU(rVal, lVal)
    accumulatedIOU+=iou
    print("      --->{}  IMAGE {} has IOU {}".format(cnter, r, iou))


print("HIT THE END SUCCESSFULLY: {}".format(cnter))
print("AVG IOU PER IMAGE: {}".format(accumulatedIOU/cnter))
    

