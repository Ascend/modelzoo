# -*- coding:utf-8 -*-
'''
Created on 2020-03-30

@author: wwx371270
'''
import sys
import commands
import os
import csv

csvroute = commands.getoutput("pwd") + "/result/"


def checkmodel(model):
    flag = 0
    if "alexnet" in model.lower():
        flag = 1
    elif "bert_nv" in model.lower():
        flag = 2
    elif "bert_nz" in model.lower():
        flag = 3
    elif "resnet50_hc" in model.lower():
        flag = 4
    elif "resnet50_nv" in model.lower():
        flag = 5
    elif "vgg16" in model.lower():
        flag = 6

    return flag


def copyrightdata():
    if checkmodel(model) == 1:
        dataroute1 = "/autotest/CI_daily/Alexnet_TF/data"
        cmd1 = "rm -rf " + dataroute1 + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/imagenet_TF/train-0000* " + dataroute1 + " huawei@123"
        commands.getoutput(cmd1)
    if checkmodel(model) == 2:
        dataroute2 = "/autotest/CI_daily/Bert_NV/data/dataset/nv-en-wiki-f16"
        cmd2 = "rm -rf " + dataroute2 + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/nv-en-wiki-f16 " + dataroute2 + " huawei@123"
        commands.getoutput(cmd2)
    if checkmodel(model) == 3:
        dataroute3 = "/autotest/CI_daily/Bert_NZ/data/bert/cn-news-128-100f"
        cmd3 = "rm -rf " + dataroute3 + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/cn-news-128-100f/cn-news-128-100f/part-0000* " + dataroute3 + " huawei@123"
        commands.getoutput(cmd3)
    if checkmodel(model) == 4:
        dataroute4 = "/autotest/CI_daily/Resnet50_HC/data/resnet50/imagenet_TF"
        cmd4 = "rm -rf " + dataroute4 + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/imagenet_TF/train-0000* " + dataroute4 + " huawei@123"
        commands.getoutput(cmd4)
    if checkmodel(model) == 5:
        dataroute5 = "/autotest/CI_daily/Resnet50_NV/data/resnet50/imagenet_TF"
        cmd5 = "rm -rf " + dataroute5 + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/imagenet_TF/train-0000* " + dataroute5 + " huawei@123"
        commands.getoutput(cmd5)
    if checkmodel(model) == 6:
        dataroute6 = "/autotest/CI_daily/VGG16_TF/data"
        cmd6 = "rm -rf " + dataroute6 + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/imagenet_TF/train-0000* " + dataroute6 + " huawei@123"
        commands.getoutput(cmd6)


# 把其他训练的数据集复制到数据集目录下
def copywrongdata(model):
    # 先复制正确的数据集，重置环境上的数据集
    copyrightdata()
    cmd = ""
    if checkmodel(model) == 1:
        dataroute = "/autotest/CI_daily/Alexnet_TF/data"
        cmd = "rm -rf " + dataroute + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/cn-wiki-128-small/* " + dataroute + " huawei@123"
    elif checkmodel(model) == 2:
        dataroute = "/autotest/CI_daily/Bert_NV/data/dataset/nv-en-wiki-f16"
        cmd = "rm -rf " + dataroute + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/imagenet_TF/train-0000* " + dataroute + " huawei@123"
    elif checkmodel(model) == 3:
        dataroute = "/autotest/CI_daily/Bert_NZ/data/bert/cn-news-128-100f"
        cmd = "rm -rf " + dataroute + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/imagenet_TF/train-0000* " + dataroute + " huawei@123"
    elif checkmodel(model) == 4:
        dataroute = "/autotest/CI_daily/Resnet50_HC/data/resnet50/imagenet_TF"
        cmd = "rm -rf " + dataroute + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/cn-wiki-128-small/* " + dataroute + " huawei@123"
    elif checkmodel(model) == 5:
        dataroute = "/autotest/CI_daily/Resnet50_NV/data/resnet50/imagenet_TF"
        cmd = "rm -rf " + dataroute + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/cn-wiki-128-small/* " + dataroute + " huawei@123"
    elif checkmodel(model) == 6:
        dataroute = "/autotest/CI_daily/VGG16_TF/data"
        cmd = "rm -rf " + dataroute + "/* && " + "cd /autotest/CI_daily && expect exp_scp.ex root@10.136.165.4:/turingDataset/cn-wiki-128-small/* " + dataroute + " huawei@123"
    commands.getoutput(cmd)


# 删除数据集文件部分内容
def deletedata(model):
    # 先复制正确的数据集，重置环境上的数据集
    copyrightdata()
    cmd1 = ""
    if checkmodel(model) == 1:
        dataroute = "/autotest/CI_daily/Alexnet_TF/data/"
        cmd1 = "ls " + dataroute
    if checkmodel(model) == 2:
        dataroute = "/autotest/CI_daily/Bert_NV/data/dataset/nv-en-wiki-f16/"
        cmd1 = "ls " + dataroute
    if checkmodel(model) == 3:
        dataroute = "/autotest/CI_daily/Bert_NZ/data/bert/cn-news-128-100f/"
        cmd1 = "ls " + dataroute
    if checkmodel(model) == 4:
        dataroute = "/autotest/CI_daily/Resnet50_HC/data/resnet50/imagenet_TF/"
        cmd1 = "ls " + dataroute
    if checkmodel(model) == 5:
        dataroute = "/autotest/CI_daily/Resnet50_NV/data/resnet50/imagenet_TF/"
        cmd1 = "ls " + dataroute
    if checkmodel(model) == 6:
        dataroute = "/autotest/CI_daily/VGG16_TF/data/"
        cmd1 = "ls " + dataroute
    data_file = commands.getoutput(cmd1).split("\n")
    for i in range(0, data_file.__len__()):
        cmd2 = "sed \'1000,$d\' -i " + dataroute + str(data_file[i])
        commands.getoutput(cmd2)


def check(model, trainlog):
    cmd1 = ""
    cmd2 = ""
    # alexnet
    if checkmodel(model) == 1:
        cmd1 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
        cmd2 = "cat " + trainlog + "|grep \"ValueError: Found no files in --data_dir matching:\""
    # bert_nv
    if checkmodel(model) == 2:
        cmd1 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
        cmd2 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
    # bert_nz
    if checkmodel(model) == 3:
        cmd1 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
        cmd2 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
    # resnet50_hc
    if checkmodel(model) == 4:
        cmd1 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
        cmd2 = "cat " + trainlog + "|grep \"ValueError: Tensor conversion requested dtype string for Tensor with dtype float32:\""
    if checkmodel(model) == 5:
        # resnet50_nv
        cmd1 = "cat " + trainlog + "|grep \"tensorflow.python.framework.errors_im pl.DataLossError:\""
        cmd2 = "cat " + trainlog + "|grep \"in parse_tfrecords_dataset\""
    # vgg16
    if checkmodel(model) == 6:
        cmd1 = "cat " + trainlog + "|grep \"Failed to get tensor data\""
        cmd2 = "cat " + trainlog + "|grep \"ValueError: Found no files in --data_dir matching:\""
    result1 = commands.getoutput(cmd1)
    result2 = commands.getoutput(cmd2)
    if result1 != "" or result2 != "":
        return 0
    else:
        return 1


if __name__ == "__main__":
    resultfile = csvroute + "errordata_report.csv"
    if os.path.exists(resultfile) == False:
        with open(resultfile, 'a+') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['train log file', 'way to change data', 'error in log'])
            f.close()
    model = sys.argv[1]
    if sys.argv[2] == "1":
        copywrongdata(model)
    elif sys.argv[2] == "2":
        deletedata(model)
    elif sys.argv[2] == "3":
        copyrightdata()
        trainroute = str(sys.argv[3])
        trainlog = trainroute + "/train_0.log"
        if check(model, trainlog) == 0:
            row = [trainlog] + [str(sys.argv[4])] + ['True']
            with open(resultfile, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                f.close()
            sys.exit(0)
        else:
            row = [trainlog] + [str(sys.argv[4])] + ['False']
            with open(resultfile, 'a+') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row)
                f.close()
            sys.exit(1)

