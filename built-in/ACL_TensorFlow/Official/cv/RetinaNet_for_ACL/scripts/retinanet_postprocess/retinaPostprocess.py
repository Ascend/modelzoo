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

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
                    6: 'train', 7: 'truck', 8: 'boat', 9: 'trafficlight', 10: 'firehydrant', 
                    11: 'stopsign', 12: 'parkingmeter', 13: 'bench', 14: 'bird', 15: 'cat', 
                    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 
                    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 
                    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 
                    31: 'snowboard', 32: 'sportsball', 33: 'kite', 34: 'baseballbat', 
                    35: 'baseballglove', 36: 'skateboard', 37: 'surfboard', 38: 'tennisracket', 39: 'bottle', 
                    40: 'wineglass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
                    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
                    51: 'carrot', 52: 'hotdog', 53: 'pizza', 54: 'donut', 55: 'cake', 
                    56: 'chair', 57: 'couch', 58: 'pottedplant', 59: 'bed', 60: 'diningtable', 
                    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
                    66: 'keyboard', 67: 'cellphone', 68: 'microwave', 69: 'oven', 
                    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
                    75: 'vase', 76: 'scissors', 77: 'teddybear', 78: 'hairdrier', 79: 'toothbrush'}

targetDir = sys.argv[1]
txtPath = sys.argv[2]
#targetDir = "result_Files/"
allFiles = os.listdir(targetDir)
#txtPath = "detections/"
outputs = []
rawScaleFile = "./rawScale.txt"
for i in allFiles:
    if i.split(".")[0][:-1] not in outputs:
        outputs.append(i.split(".")[0][:-1])

print("Number of files to be handled : {}".format(len(outputs)))
print(outputs)

def readAsFP16(f):
    return np.fromfile(targetDir+f, np.float16)

def writeResult(path, filename, b, s, l):
    file_to_be_write = open(path+filename.split('_')[1]+'.txt', "w")
    for box, score, label in zip(b, s, l):
        # scores are sorted so we can break
        if score < 0.5:
            break
        file_to_be_write.write(labels_to_names[label]+" "+str(score)+" "+
                        str(int(box[0]))+" "+str(int(box[1]))+" "+str(int(box[2])-int(box[0]))+" "+str(int(box[3])-int(box[1]))+"\n") 
    file_to_be_write.close()

def readRawScale(rawScaleTxt):
    with open(rawScaleTxt) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    lineDict = {}
    for line in content:
        lineSpl = line.split(" ")
        lineDict[lineSpl[0].split(".")[0]]=[float(lineSpl[1]), float(lineSpl[2])]
    return lineDict

def getBox(raw, x, y):
    cnter = 0
    res = []
    while cnter<len(raw):
        singleBbox = [raw[cnter]/x, raw[cnter+1]/y, raw[cnter+2]/x, raw[cnter+3]/y]
        res.append(singleBbox)        
        cnter+=4
    return res

dic = readRawScale(rawScaleFile)

for cnt,f in enumerate(outputs):
    print("{} ---> ".format(cnt)+f)
    rawBoxes = readAsFP16(f+"0.bin")
    scores = readAsFP16(f+"1.bin")
    classes = readAsFP16(f+"2.bin")
    boxes = getBox(rawBoxes, dic[f.split('_')[1]][0], dic[f.split('_')[1]][1])
    writeResult(txtPath, f, boxes, scores, classes)

    '''
    for k,j in enumerate(scores):
        if j<0.5:
            continue
        print("Score:\t"+str(j)+"\t"+"Class: "+"\t"+str(classes[k])+"-"+labels_to_names[int(classes[k])]+"\t"+getBox(k, boxes))
    '''
