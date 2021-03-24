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


def evaluating_op(log, label, num_classes=2):
    predict = np.argmax(log, axis=-1)
    # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


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
    # print("start bin data")
    binDataExist = False
    start = 0
    output_path = output_path + "/"
    # 判断路径 创建路径
    # if clear:
    #     os.system("rm -rf "+output_path+"data")
    # if os.path.isdir(output_path+"data"):
    #     # binDataExist = True
    #     start = len(os.listdir(output_path+"data"))
    # else:
    #     os.makedirs(output_path+"data")

    # if os.path.isdir(output_path+"label"):
    #     pass
    # else:
    #     os.makedirs(output_path+"label")

    imgs, imageLabel = load_h5(data_path)
    # imgs = np.array(imgs)
    imageLabel = np.array(imageLabel)
    # imgs = dimg2data(imgs)
    # imgs = imgs.astype(np.float32)
    imageLabel = label2data(imageLabel)
    # inference = []
    # count = 0
    # for i in range(0, len(imageLabel[:testNumber]) // batch):
        # images = imgs[i*batch:batch*(i+1)]
        # images.astype(np.float32)
        # images.tofile(output_path+"data/"+str(i)+".bin")
        # count += 1
    # left = len(imageLabel) - batch*count
    # add = batch - left
    # images = imgs[batch*count:]
    # images = np.concatenate([images, imgs[:add]], axis=0)
    # images.astype(np.float32)
    # images.tofile(output_path+"data/"+str(count)+".bin")
    # print("[INFO]    images have been converted to bin files")
    return imageLabel


def clear_files(output_path):
    os.system("rm -rf %sdata" % output_path)
    os.makedirs(output_path+"data")


def msamePath(output_path, inference_path, model_path):
    """
    使用文件夹推理
    """
    if os.path.isdir(inference_path):
        os.system("rm -rf "+inference_path)
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + "data"
    print("./msame --model "+model_path + " --input "+output_path +
          " --output "+inference_path + " --outfmt BIN")
    os.system("./msame --model "+model_path + " --input " +
              output_path + " --output "+inference_path + " --outfmt BIN")
    print("[INFO]    推理结果生成结束")


def segmentation_cls_inference_files(inference_path, sup_labels):
    # 获得这个文件夹下面所有的bin 然后排序每个读进去 就行
    output_num = 0
    h, w, c = 256, 320, 2
    files = len(os.listdir(inference_path))
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    lastfile = files[-1]
    res = []
    left = len(imageLabel) % batch
    # sup_labels = np.concatenate([sup_labels,sup_labels[:left+1]])
    print(left)
    for file in files:
        if file.endswith(".bin"):
            tmp = np.fromfile(file, dtype='float32')
            tmpLength = len(tmp)
            tmp = tmp.reshape(batch, h, w, c)
            if file == lastfile:
                tmp = tmp[:left]
                acc = evaluating_op(
                    tmp, sup_labels[(output_num)*batch:])
            else:
                acc = evaluating_op(
                    tmp, sup_labels[output_num*batch:(output_num+1)*batch])
                output_num += 1
            res.append(acc)
    print(">>>>> ", "共 %d 测试样本 \t" % (output_num*batch+left),
          "accuracy:%.6f" % (sum(res) / len(res)))


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    inference_path = sys.argv[3]
    model_path = sys.argv[4]  # model的地址
    imageLabel = gen_image_bin(data_path, output_path)
    msamePath(output_path, inference_path, model_path)
    segmentation_cls_inference_files(inference_path, imageLabel)
