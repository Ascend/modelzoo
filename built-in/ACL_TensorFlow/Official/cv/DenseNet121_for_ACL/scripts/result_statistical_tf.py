from pprint import pprint
import os
import os
import sys
import json
import numpy as np

def read_file(file_path, csv_file_path):
    with open(file_path, 'r') as f:
        res = f.readlines()
        r = enumerate(res)
        list = sorted(r, key=lambda x: float(x[1]), reverse=True)[0:5]
        print(list)
    file_name = file_path.split("/")[-1]
    #print(file_name)
    name_list = file_name.split("_")
    # print(name_list)
    # img_name = name_list[1] + "_" + name_list[2] + "_" + name_list[3]
    img_name = name_list[1] + "_" + name_list[2] + "_" + name_list[3]
    #print(img_name)
    with open(csv_file_path, 'r') as cs:
        rs_list = cs.readlines()
        for name in rs_list:
            if img_name in str(name):
                num = str(name).split(" ")[1]
                print(num)
                break
    p = 0
    for t in list:
        if (t[0])  == int(num) and list.index(t) == 0 and float(t[1]) != 0:
            #print("<%s>----------top1,top5---%s" % (img_name, t[1]))
            res = "top1,top5"
            if float(t[1]) >= 0.9:
                p = 1
            break
        elif (t[0])== int(num) and list.index(t) != 0 and float(t[1]) != 0:
            #print("<%s>----------top5--------%s" % (img_name, t[1]))
            res = "top5"
            if float(t[1]) >= 0.9:
                p = 1
            break
        elif (t[0])!= int(num) and list.index(t) == 4:
            #print("<%s>----------not find" % (img_name))
            res = ""
    return res, p


# read_file("D:/11/image_file/davinci_ILSVRC2012_val_00000003_output_0_prob_0.txt", "./JPG.csv")


def read_dirt_file_name(direct_path, csv_file_path):
    files_list = os.listdir(direct_path)
    file_name_list = []
    # print(files_list)
    for file_name in files_list:
        if ".txt" in file_name:
            file_name_list.append(file_name)
    # print(len(file_name_list), file_name_list)
    #print("file_name_list:" ,len(file_name_list))
    t1 = 0
    t5 = 0
    p1_num = 0
    for f_n in file_name_list:
        file_path = direct_path + "/" + f_n
        print(file_path)
        res, p = read_file(file_path, csv_file_path)
        # print(res, p)
        if p == 1:
            p1_num += 1
        if res == "top1,top5":
            t1 += 1
        if res == "top5" or res == "top1,top5":
            t5 += 1
    print("top1_num = %d, top5_num = %d, above_90_num = %d filename = %s\n" % (t1, t5, p1_num, f_n))
    img_num = len(file_name_list)
   
    top1_accuracy_rate = '%.6f' % (float(t1) / float(img_num))
    top5_accuracy_rate = '%.6f' %(float(t5) / float(img_num))
    ntopn_avove90_rate = '%.6f' %(float(p1_num) / float(img_num))
  
    print("img_num %s top1_accuracy_rate %s top5_accuracy_rate %s ntopn_avove90_rate %s" % (img_num, top1_accuracy_rate, top5_accuracy_rate, ntopn_avove90_rate))
    "write to file"
    # os.mknod("/home/yaoaijuan/ret/App_classify_002/scripts/result.txt")
    with open("./result.txt", 'w+') as f:
        f.write("img_num %s top1_accuracy_rate %s top5_accuracy_rate %s ntopn_avove90_rate %s" % (img_num, top1_accuracy_rate, top5_accuracy_rate, ntopn_avove90_rate))

if __name__ == '__main__':

    folder_davinci_target = sys.argv[1]  # davinci*.txt file path
    true_file_path = sys.argv[2]  # True value file

    if not (os.path.exists(folder_davinci_target)):
        print("Davinci target file folder does not exist.")

    if not (os.path.exists(true_file_path)):
        print("Ground truth file does not exist.")

    print("read_dirt_file_name start.")

    read_dirt_file_name(folder_davinci_target, true_file_path)
