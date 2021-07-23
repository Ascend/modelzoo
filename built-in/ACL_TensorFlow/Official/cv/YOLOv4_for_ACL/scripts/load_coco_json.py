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

import time as time
import json
import numpy as np
from collections import defaultdict
import cv2 as cv
import os
import numpy as np

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
            print(ids)
            print()
        return list(ids)
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

#coco数据集解析出来的ID与预测时ID不同
labels_to_names = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 
                    7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight', 11: 'firehydrant', 
                    13: 'stopsign', 14: 'parkingmeter', 15: 'bench', 16: 'bird', 17: 'cat', 
                    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 
                    23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 
                    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 
                    36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat', 
                    40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket', 44: 'bottle', 
                    46: 'wineglass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 
                    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 
                    57: 'carrot', 58: 'hotdog', 59: 'pizza', 60: 'donut', 61: 'cake', 
                    62: 'chair', 63: 'couch', 64: 'pottedplant', 65: 'bed', 67: 'diningtable', 
                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 
                    76: 'keyboard', 77: 'cellphone', 78: 'microwave', 79: 'oven', 
                    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 
                    86: 'vase', 87: 'scissors', 88: 'teddybear', 89: 'hairdrier', 90: 'toothbrush'}

annFile = './instances_val2017.json'
coco = COCO(annFile)
anns = coco.imgToAnns
print(len(anns))
gt_dir = './yolov4_postprocess/groundtruths/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for i in sorted(anns):
    filename = str(i).zfill(12)+".txt"
    print(filename)
    content = ""
    for j in anns[i]:
        content += labels_to_names[j['category_id']] + ' '
        for xy in j['bbox']:
            content += str(int(xy)) + ' '
        content += '\n'

    with open(os.path.join(gt_dir,filename),'w') as f:
        f.write(content)
