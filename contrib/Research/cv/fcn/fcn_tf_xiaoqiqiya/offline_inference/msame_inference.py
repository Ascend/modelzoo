import sys
import os
import numpy as np
import cv2
import json
import time

batch = 1
clear = False
allNum = 736


def get_result(confusion_matrix):
    # pixel accuracy
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    # mean iou
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def getLabel(data_path, output_path):
    hwlist = np.load(output_path + "/label" + "/hwlist.npy")
    label = np.load(output_path + "/label" + "/label.npy")
    return (hwlist, label)


def clear_files(output_path):
    os.system("rm -rf %sdata" % output_path)
    os.makedirs(output_path+"data")


def get_matrix(log, label, num_classes=21):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


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
    print(inference_path)
    print("[INFO]    推理结果生成结束")


def segmentation_cls_inference_files(inference_path, sup_labels):
    # 获得这个文件夹下面所有的bin 然后排序每个读进去 就行
    output_num = 0
    oh, ow, c = 520, 520, 21
    hwlist, label = sup_labels
    files = len(os.listdir(inference_path))
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    c_matrix = np.zeros((21, 21))
    for file in files:
        if file.endswith(".bin"):
            # =================================== #
            h, w, y_in = hwlist[output_num][0], hwlist[output_num][1], label[output_num]
            # =================================== #
            tmp = np.fromfile(file, dtype='float32')
            tmpLength = len(tmp)
            tmp = tmp.reshape(batch, oh, ow, c)
            pred = tmp[:, 4:-4, 4:-4, :]
            pred = pred[0, (512 - h[0]) // 2: (512 - h[0]) // 2 +
                        h[0], (512 - w[0]) // 2: (512 - w[0]) // 2 + w[0]]
            y_in = y_in[0, (512 - h[0]) // 2: (512 - h[0]) // 2 +
                        h[0], (512 - w[0]) // 2: (512 - w[0]) // 2 + w[0]]
            pre = np.argmax(pred, axis=-1)
            c_matrix += get_matrix(pred, y_in, num_classes=21)
            output_num += 1
    res = get_result(c_matrix)
    print(">>>>> ", "共 %d 测试样本 \t" % (output_num*batch),
          "MIoU: %.6f" % (res))
    assert output_num*batch == allNum


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    inference_path = sys.argv[3]
    model_path = sys.argv[4]  # model的地址
    imageLabel = getLabel(data_path, output_path)
    msamePath(output_path, inference_path, model_path)
    segmentation_cls_inference_files(inference_path, imageLabel)
