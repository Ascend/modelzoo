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

# -*- coding: utf-8 -*-
import numpy as np
import argparse
import sys
import json
from PIL import Image
from os.path import join
import cv2
from matplotlib import pyplot as plt

# 设标签宽W，长H
def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    hist = np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def compute_mIoU(gt_dir, pred_dir, devkit_dir):  # 计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and
    """
    num_classes = 21
    print('Num classes', num_classes)
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')  # 在这里打开记录分割图片名称的txt
    label_path_list = join(devkit_dir, 'val.txt')  # ground truth和自己的分割结果txt一样
    gt_imgs = open(label_path_list, 'r').read().splitlines()  # 获得验证集标签名称列表
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = open(image_path_list, 'r').read().splitlines()  # 获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, 'davinci_{}_output0.bin'.format(x)) for x in pred_imgs]

    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        pred = np.fromfile(pred_imgs[ind], np.float32).astype(np.uint8).reshape(513, 513)  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind] + '.png').resize((513,513)),np.uint8)  # 读取一张对应的标签，转化成numpy数组
        if len(label.flatten()) != len(pred.flatten()):
            lab_shape = label.shape
            pred = cv2.resize(pred, label.shape, interpolation=cv2.INTER_NEAREST)

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加

    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值
        print('{: <15}:{}'.format(name_classes[ind_class], round(mIoUs[ind_class] * 100, 2)))
    print('Overall mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs

if __name__ == '__main__':
    root_dir = sys.argv[1]
    gt_dir = join(root_dir,'SegmentationClass')
    list_dir = join(root_dir,'ImageSets/Segmentation/')
    pred_dir = sys.argv[2]
    result = compute_mIoU(gt_dir,pred_dir,list_dir)

