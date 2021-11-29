# assemble infer result 
# -*- coding: utf-8 -*-
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import codecs
import argparse
import evaluation_utils
import numpy as np


parser = argparse.ArgumentParser(description='Calculate the BLEU score')
parser.add_argument('--source_dir', type=str,  default='../result_Files', help='infer result folder')
parser.add_argument('--assemble_file', type=str, default='../result_Files/infer_assemble', help='vocable file')


def assemble_infer_result(source_dir, target_file):
    res = []
    print("====len listdir:", len(os.listdir(source_dir)))
    file_seg_str = "_".join(os.listdir(source_dir)[0].split("_")[:-2])
    print("====file_seg_str:", file_seg_str)

    with open(target_file, "w") as f:
        for i in range(len(os.listdir(source_dir))):
            file = file_seg_str + "_" + str(i) + "_output0.bin"
            file_full_name = os.path.join(source_dir, file)
            val = list(np.fromfile(file_full_name, np.int32).reshape((80)))
            j = ""
            for i in val:
                j = j + str(i) + " "

            print("===val", j)
            print("====val type:", type(val))
            f.write(j + "\n")
    print("Program hit the end successfully")


if __name__ == "__main__":
    args = parser.parse_args()
    source_dir = args.source_dir
    target_file = args.assemble_file
    assemble_infer_result(source_dir, target_file)
    


