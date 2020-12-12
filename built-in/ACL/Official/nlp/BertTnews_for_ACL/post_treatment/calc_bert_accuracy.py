#
# Copyright 2020 Huawei Technologies Co., Ltd
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import numpy as np


def read_to_json(input_file):
	with open(input_file, "r") as input_f:
		reader = input_f.readlines()
		lines = []
		for line in reader:
			lines.append(json.loads(line.strip()))
		return lines


def get_real_label(input_file):
	examples_lines = read_to_json(input_file)
	real_label_dict = {}
	for i, line in enumerate(examples_lines):
		label_id = str(line['label'])
		real_label_dict[str(i)] = label_id
	return real_label_dict


def get_lable_list(label_list_file):
	labels_lines = read_to_json(label_list_file)
	label_list = []
	for i, line in enumerate(labels_lines):
		label_id = str(line['label'])
		label_list.append(label_id)
	return label_list


def calc_bert_infer_accuracy(predic_out_folder, real_file, label_list_file):
	label_list = get_lable_list(label_list_file)
	print("label_list:", label_list)
	index2label_map = {}
	for (i, label) in enumerate(label_list):
		index2label_map[i] = label
	print("index2label_map:", index2label_map)
	real_label_dict = get_real_label(real_file)
	predict_cnt = 0
	acc_cnt = 0
	for foldername, subfolders, filenames in os.walk(predic_out_folder):
		for filename in filenames:
			if filename.endswith(".bin") == False:
				continue
			id = filename.split("_")[3]
			predict_result_np_array = np.fromfile(os.path.join(predic_out_folder, filename), 
						dtype=np.float32)
			predict_result_list = predict_result_np_array.tolist()
			label_index = predict_result_list.index(max(predict_result_list))
			actural_label = index2label_map[label_index]

			real_label = real_label_dict[str(id)]
			print("******id:", id, "act_label:", actural_label, "real_label:", real_label)
			predict_cnt = predict_cnt + 1
			if actural_label == real_label:
				acc_cnt = acc_cnt + 1
	print("---------predict_cnt:", predict_cnt, "acc_cnt", acc_cnt)
	predict_accuracy = 0.0
	if predict_cnt > 0 :
		predict_accuracy = acc_cnt / predict_cnt
		print("---------predict_accuracy:", predict_accuracy)

	current_path = os.getcwd()
	result_save_file = current_path + "/../output/predict_accuracy.txt"
	fp = open(result_save_file, "w")
	fp.write("predict_cnt: %d, correct_cnt: %d\n" % (predict_cnt, acc_cnt))
	fp.write("predict_accuracy: %0.4f\n" % predict_accuracy )
	fp.close()


if __name__ == "__main__":
	predict_out_folder = sys.argv[1]
	real_file = sys.argv[2]
	label_list_file = sys.argv[3]
	calc_bert_infer_accuracy(predict_out_folder, real_file, label_list_file)

