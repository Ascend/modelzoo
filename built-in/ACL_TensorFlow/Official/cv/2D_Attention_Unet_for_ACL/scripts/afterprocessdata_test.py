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
import time, datetime
import os, sys
import argparse
import cv2, math
import numpy as np
import utils
import csv
from utils import load_divided

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='lashan', help='Dataset you are using')
parser.add_argument('--crop_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--dataset', type=str, default='lashan', help='Dataset you are using')
parser.add_argument('--benchmark_path', type=str, default='lashan', help='benchmark path')
args = parser.parse_args()

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    h, w = args.crop_height, args.crop_width
    image = cv2.resize(image, (h, w))
    return image

def prepare_data(dataset_dir=args.dataset):
    val_input_names, val_output_names, test_input_names, test_output_names = load_divided(dataset_dir)

    print("validation data length: {}".format(len(test_input_names)))
    val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return val_input_names, val_output_names, test_input_names, test_output_names


#Takes an absolute  file path and returns the name of the file without th extension

def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

filenames = os.listdir(args.path)

sort_num_first = []
for file_ in filenames:
    if file_.endswith(".bin"):
        sort_num_first.append(int(file_.split("_")[1]))
        sort_num_first.sort()
sorted_file = []

for sort_num in sort_num_first:
    for file_ in filenames:
        if str(sort_num) == file_.split("_")[1] and file_.endswith(".bin"):
            sorted_file.append(file_)
print(sorted_file)
(class_names_list, label_values) = utils.get_label_info(os.path.join(args.dataset, 'class_dict.csv'))
class_names_string = ''
for class_name in class_names_list:
    if (not (class_name == class_names_list[(- 1)])):
        class_names_string = ((class_names_string + class_name) + ',')
    else:
        class_names_string = (class_names_string + class_name)
target = open(('%s/test_scores.csv' %args.benchmark_path), 'w')
target.write(('val_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n' % class_names_string))
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []
num_classes = len(label_values)
print('Num Classes -->',num_classes)

val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()

for i in range(len(sorted_file)):
    print(test_output_names[i])
    gt = load_image(test_output_names[i])[:args.crop_height, :args.crop_width]
    gt = utils.reverse_one_hot(utils.one_hot_it(gt, label_values))
    cwd = os.getcwd()
    num = []
    print(cwd + "/" + args.path + sorted_file[i])
    num = np.fromfile(cwd + "/" + args.path + sorted_file[i], dtype='float32')
    print(num)

    output_image = np.reshape(num, (1, 224, 224, 2))

    output_image = np.array(output_image[0, :, :, :])
    output_image = utils.reverse_one_hot(output_image)
    out_vis_image = utils.colour_code_segmentation(output_image, label_values)
    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt,num_classes=num_classes)
    file_name = filepath_to_name(test_input_names[i])
    target.write("%s, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou))
    for item in class_accuracies:
        target.write(", %f" % (item))
    target.write("\n")
    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)

    gt = utils.colour_code_segmentation(gt, label_values)
    im = cv2.imread(test_input_names[i])
    cv2.imwrite("%s/%s_pred.png" % ("Val", file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_img.png" % ("Val", file_name),im)
    cv2.imwrite("%s/%s_gt.png" % ("Val", file_name), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list,axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
target.write("%s, %f, %f, %f, %f, %f" %(file_name, accuracy, prec, rec, f1, iou))
target.close()

print("Average test accuracy = ",avg_score)
print("Average per class test accuracies\n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f " % (class_names_list[index], item))
print("Average precision = ", avg_precision)
print("Average recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average mean IoU score = ", avg_iou)
