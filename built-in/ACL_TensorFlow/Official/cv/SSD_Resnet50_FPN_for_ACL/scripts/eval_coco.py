#-*- coding:utf-8 -*-
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

# import matplotlib.pyplot as plt
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import numpy as np 
import pylab,json
import sys
# pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def get_img_id(file_name): 
    ls = [] 
    myset = [] 
    annos = json.load(open(file_name, 'r')) 
    for anno in annos: 
      ls.append(anno['image_id']) 
    myset = {}.fromkeys(ls).keys() 
    return myset


if __name__ == '__main__': 
    annType = ['segm', 'bbox', 'keypoints']#set iouType to 'segm', 'bbox' or 'keypoints'
    annType = annType[1] # specify type here
    cocoDt_file = sys.argv[1]
    cocoGt_file = sys.argv[2]
    cocoGt = COCO(cocoGt_file)#取得标注集中coco json对象

    imgIds = get_img_id(cocoDt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)#取得结果集中image json对象
    imgIds = sorted(imgIds)#按顺序排列coco标注集image_id
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds#参数设置
    cocoEval.evaluate()#评价
    cocoEval.accumulate()#积累
    cocoEval.summarize()#总结
