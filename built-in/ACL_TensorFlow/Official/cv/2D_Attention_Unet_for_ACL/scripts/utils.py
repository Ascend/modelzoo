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
import random
import time,datetime
import os,sys
import cv2, math
import numpy as np
import csv

def load_divided(dataset_dir):
    val_input_names = []
    val_output_names = []
    test_input_names = []
    test_output_names = []
    cwd = os.getcwd()
    for file in os.listdir(dataset_dir + "/val"):
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    return val_input_names, val_output_names, test_input_names, test_output_names

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is 
    the classified class key.

    #Arguments
        image: The one-hot format image

    #Returns
        A 2D array with the same width and height as the input, but 
        with a depth size of 1, where each pixel value is the calssified 
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x

def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    #Arguments
        label: The 2D array segmentation image label
        label_values

    #Returns
        A 2D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    #Arguments
        image: single channel array where each value represents the class key.
        label_values

    #Returns
        Colour coded image for segmentation visualization
    """

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in csv format!

    #Arguments
        csv_path: The file path of the class dictionairy

    #Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")
    class_names = []
    label_values = []
    
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values

def compute_global_accuracy(pred, label):
    total =  len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] ==0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])
    return accuracies

from sklearn.metrics import precision_score, \
        recall_score, confusion_matrix, classification_report, \
        accuracy_score, f1_score

def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou
def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred,flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)
    return global_accuracy, class_accuracies, prec, rec, f1, iou
