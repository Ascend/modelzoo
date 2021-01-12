import sys
import os
import numpy as np
import cv2
import preprocess
import json
import time

############
#  4个输入
#  输入1 datapath test文件的位置
#  输入2 output_path 是bin文件输出位置
#  输入3 inference_path 是 inference的位置
#  输入4 modelpath 是 model的路径
###########
batch = 1
clear = True


def gen_cifar100_np_bin(data_path, output_path, testNumber=100):
    output_path = output_path + "/"
    print("[info]  begin to generate bin file")
    if clear:
        os.system("rm -rf "+output_path+"data")
    if os.path.isdir(output_path+"data"):
        pass
    else:
        os.makedirs(output_path+"data")

    if os.path.isdir(output_path+"label"):
        pass
    else:
        os.makedirs(output_path+"label")
    imgs = np.load(data_path).reshape(-1, 32, 32, 3)
    label_path = "/".join(data_path.split("/")[:-1])+"/cifar100_label.npy"
    imageLabel = np.load(label_path).reshape(-1)
    count = 0
    testNumber = len(imageLabel) if testNumber == 100 else testNumber
    print(testNumber)
    for i in range(0, testNumber // batch):
        images = imgs[i*batch:(i+1)*batch]
        images.astype(np.float32)
        images.tofile(output_path+"data/"+str(count)+".bin")
        # print(output_path+"data/"+str(count)+".bin")
        count += 1
    print("[INFO]    images have been converted to bin files")
    return imageLabel


def clear_files(output_path):
    os.system("rm -rf %sdata" % output_path)
    os.makedirs(output_path+"data")


def parse_label(output_path):
    label_path = output_path+"/label/imageLabel.npy"
    sup_labels = np.load(label_path)
    # 读到label的path  然后加载label
    return sup_labels


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


def iterator_cls_inference_files(inference_path, sup_labels):
    """
    通过文件夹推理
    """
    # 获得这个文件夹下面所有的bin 然后排序每个读进去 就行
    output_num = 0
    files = len(os.listdir(inference_path))
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    res = []
    for file in files:
        if file.endswith(".bin"):
            tmp = np.fromfile(file, dtype='float32')
            tmpLength = len(tmp)
            tmp.resize(batch, tmpLength // batch)
            inf_label = np.argmax(tmp, axis=1)
            for i in range(batch):
                res.append(1 if inf_label[i] == sup_labels[output_num] else 0)
                output_num += 1
    print(">>>>> ", "共 %d 测试样本 \t" % (output_num),
          "accuracy:%.6f" % (sum(res) / len(res)))
    return sum(res), len(res)


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    inference_path = sys.argv[3]
    model_path = sys.argv[4]  # model的地址
    imageLabel = gen_cifar100_np_bin(data_path, output_path)
    msamePath(output_path, inference_path, model_path)
    iterator_cls_inference_files(inference_path, imageLabel)
