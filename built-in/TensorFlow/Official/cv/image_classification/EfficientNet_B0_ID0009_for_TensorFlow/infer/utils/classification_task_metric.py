# coding = utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import numpy as np
import time

np.set_printoptions(threshold=sys.maxsize)

LABEL_FILE = "HiAI_label.json"


def gen_file_name(img_name):
    """
    :param img_name: file path of image
    :return: filename
    """
    full_name = img_name.split('/')[-1]
    index = full_name.rfind('.')
    return full_name[:index]


def cre_groundtruth_dict(gtfile_path):
    """
    :param gtfile_path: file contains the image name and label number
    :return: dictionary key image name, value is label number
    """
    img_gt_dict = {}
    for gt_file in os.listdir(gtfile_path):
        if gt_file != LABEL_FILE:
            with open(os.path.join(gtfile_path, gt_file), 'r') as f:
                gt = json.load(f)
                ret = gt["image"]["annotations"][0]["category_id"]
                img_gt_dict[gen_file_name(gt_file)] = ret
    return img_gt_dict


def cre_groundtruth_dict_fromtxt(gtfile_path):
    """
    :param gtfile_path: file contains the image name and label number
    :return: dictionary key image name, value is label number
    """
    img_gt_dict = {}
    with open(gtfile_path, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            img_name = temp[0].split(".")[0]
            img_lab = temp[1]
            img_gt_dict[img_name] = img_lab
    return img_gt_dict


def load_statistical_predict_result(filepath):
    """
    :function: the prediction result file data extraction
    :param filepath: result filepath
    :return: probabilities, number of label, in_type, color
            n_label: number of label
            data_vec: the probabilities of prediction in the 1000
    """
    with open(filepath, 'r') as f:
        data = f.readline()
        temp = data.strip().split(" ")
        n_label = len(temp)
        data_vec = np.zeros(n_label, dtype=np.float32)
        in_type = ''
        color = ''
        if n_label == 0:
            in_type = f.readline()
            color = f.readline()
        else:
            for ind, cls_ind in enumerate(temp):
                data_vec[ind] = np.int(cls_ind)
    return data_vec, n_label, in_type, color


def create_visualization_statistical_result(prediction_file_path,
                                            result_store_path, json_file,
                                            img_gt_dict, topn=5):
    """
    :param prediction_file_path: input file path of model prediction
    :param result_store_path: restore path of result
    :param json_file: save the accuracy of topN
    :param img_gt_dict: dict of images' ground true label
    :param topn: calculate topN accuracy
    :return: result file
    """
    writer = open(os.path.join(result_store_path, json_file), 'w')
    table_dict = dict()
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []

    count = 0
    res_cnt = 0
    n_labels = ""
    count_hit = np.zeros(topn)
    for tfile_name in os.listdir(prediction_file_path):
        count += 1
        temp = tfile_name.split('.')[0]
        index = temp.rfind('_')
        img_name = temp[:index]
        filepath = os.path.join(prediction_file_path, tfile_name)

        ret = load_statistical_predict_result(filepath)
        prediction = ret[0]
        n_labels = ret[1]

        gt = img_gt_dict[img_name]
        if n_labels == 1000:
            real_label = int(gt)
        elif n_labels == 1001:
            real_label = int(gt) + 1
        else:
            real_label = int(gt)

        res_cnt = min(len(prediction), topn)
        for i in range(res_cnt):
            if str(real_label) == str(int(prediction[i])):
                count_hit[i] += 1
                break
    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(n_labels)}])
        if count == 0:
            accuracy = 0
        else:
            accuracy = np.cumsum(count_hit) / count
        for i in range(res_cnt):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(
                                            round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer)
    writer.close()


if __name__ == '__main__':
    start = time.time()
    try:
        # txt file path
        folder_davinci_target = sys.argv[1]
        # annotation files path, "val_label.txt"
        annotation_file_path = sys.argv[2]
        # the path to store the results json path
        result_json_path = sys.argv[3]
        # result json file name
        json_file_name = sys.argv[4]
    except IndexError:
        print("Please input arguments: argv[1] is the predict dir, "
              "argv[2] is the label file, argv[3] is the output dir, "
              "argv[4] is the output json file. the command such as: "
              "python classification_task_metric.py result/ ./val_label.txt ./ result.json")
        exit(1)

    if not os.path.exists(folder_davinci_target):
        print("target file folder does not exist.")

    if not os.path.exists(annotation_file_path):
        print("Ground truth file does not exist.")

    if not os.path.exists(result_json_path):
        print("Result folder doesn't exist.")

    img_label_dict = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(folder_davinci_target,
                                            result_json_path, json_file_name,
                                            img_label_dict, topn=5)

    elapsed = time.time() - start
