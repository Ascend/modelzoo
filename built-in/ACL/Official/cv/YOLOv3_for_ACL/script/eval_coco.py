#-*- coding:utf-8 -*-
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


'''
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.317
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.625
'''

if __name__ == '__main__': 
    annType = ['segm', 'bbox', 'keypoints']#set iouType to 'segm', 'bbox' or 'keypoints'
    annType = annType[1] # specify type here
    #cocoGt_file = '/autotest/c00314821/yolov3/data/coco/coco2014/annotations/instances_val2014.json'
     
    # print(list(cocoGt.anns.items())[:10])
    # print(cocoGt.anns[318219])
    # input()
    # cocoDt_file = 'result.json'
    cocoDt_file = sys.argv[1]
    cocoGt_file = sys.argv[2]
    cocoGt = COCO(cocoGt_file)#取得标注集中coco json对象

    imgIds = get_img_id(cocoDt_file) 
    # print(len(imgIds))
    cocoDt = cocoGt.loadRes(cocoDt_file)#取得结果集中image json对象
    imgIds = sorted(imgIds)#按顺序排列coco标注集image_id
    # print(imgIds)
    # input()
    # imgIds = imgIds[0:5000]#标注集中的image数据
    cocoEval = COCOeval(cocoGt, cocoDt, annType) 
    cocoEval.params.imgIds = imgIds#参数设置
    cocoEval.evaluate()#评价
    cocoEval.accumulate()#积累
    cocoEval.summarize()#总结
