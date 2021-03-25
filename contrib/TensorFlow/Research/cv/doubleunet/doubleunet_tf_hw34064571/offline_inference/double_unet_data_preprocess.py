import sys
import os
import numpy as np
import cv2
import preprocess
import json
import time
import h5py

batch = 24
clear = True


def load_h5(path):
    print('loading', path)
    file = h5py.File(name=path, mode='r')
    return file['images'], file['labels']


def label2data(y):
    if np.max(y) > 2:
        y = y / 255
    return y


def dimg2data(x):
    x = x / 255
    return x

def gen_image_bin(data_path, output_path, testNumber=100):
    """
    生成数据所需要的bin文件
    """
    print("start bin data")
    binDataExist = False
    start = 0
    output_path = output_path + "/"
    # 判断路径 创建路径
    if clear:
        os.system("rm -rf "+output_path+"data")
    if os.path.isdir(output_path+"data"):
        # binDataExist = True
        start = len(os.listdir(output_path+"data"))
    else:
        os.makedirs(output_path+"data")

    if os.path.isdir(output_path+"label"):
        pass
    else:
        os.makedirs(output_path+"label")

    imgs, imageLabel = load_h5(data_path)
    imgs = np.array(imgs)
    imageLabel = np.array(imageLabel)
    imgs = dimg2data(imgs)
    imgs = imgs.astype(np.float32)
    imageLabel = label2data(imageLabel)
    inference = []
    count = 0
    for i in range(0, len(imageLabel[:testNumber]) // batch):
        images = imgs[i*batch:batch*(i+1)]
        images.astype(np.float32)
        images.tofile(output_path+"data/"+str(i)+".bin")
        count += 1
    left = len(imageLabel) - batch*count
    add = batch - left
    images = imgs[batch*count:]
    images = np.concatenate([images, imgs[:add]], axis=0)
    images.astype(np.float32)
    images.tofile(output_path+"data/"+str(count)+".bin")
    print("[INFO]    images have been converted to bin files")
    return imageLabel

def clear_files(output_path):
    os.system("rm -rf %sdata" % output_path)
    os.makedirs(output_path+"data")


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    imageLabel = gen_image_bin(data_path, output_path)
