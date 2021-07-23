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

import numpy as np
import os
import time
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_result", type=str, default="../../result_Files")
    parser.add_argument("--label", type=str, default="../data/input_50000.csv")
    parser.add_argument("--output_index", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dtype", type=str, default='float32')   #datatype of bin files
    args = parser.parse_args()

    image_cnt = 0
    top1_cnt = 0
    top5_cnt = 0
    ground_truth={}
    if args.label.endswith(".csv"):
        with open(args.label, 'r') as cs:
            rs_list = cs.readlines()
            for line in rs_list:
                image_name = line.split(',')[0].split('.JPEG')[0]
                label = int(line.split(',')[1])
                label += args.offset
                ground_truth[image_name]=label
    elif args.label.endswith(".txt"):
        with open(args.label, 'r') as cs:
            rs_list = cs.readlines()
            for line in rs_list:
                image_name = line.split(' ')[0].split('.JPEG')[0]
                label = int(line.split(' ')[1].replace("\n",""))
                label += args.offset
                ground_truth[image_name]=label

    for i in sorted(ground_truth):
        try:
            image_name = i
            label = ground_truth[i]
            #查看输出的文件
            if os.path.exists(os.path.join(args.infer_result,'davinci_{}_output{}.bin'.format(image_name,args.output_index))):
                bin_path = os.path.join(args.infer_result,'davinci_{}_output{}.bin'.format(image_name, args.output_index))
                pred = np.fromfile(bin_path, dtype=args.dtype)
            elif os.path.exists(os.path.join(args.infer_result,'davinci_{}.JPEG_output{}.bin'.format(image_name, args.output_index))):
                bin_path = os.path.join(args.infer_result,'davinci_{}.JPEG_output{}.bin'.format(image_name, args.output_index))
                pred = np.fromfile(bin_path, dtype=args.dtype)
            elif os.path.exists(os.path.join(args.infer_result,'{}_output_{}.bin'.format(image_name,args.output_index))):
                bin_path = os.path.join(args.infer_result,'{}_output_{}.bin'.format(image_name, args.output_index))
                pred = np.fromfile(bin_path, dtype=args.dtype)
            else:
                continue
            top1=np.argmax(pred)
            if label == top1:
                top1_cnt += 1
            if label in np.argsort(-pred)[0:5]:
                top5_cnt += 1
            image_cnt+=1
            print("{}, gt label:{: >4}, predict results:{}".format(image_name,label,str(np.argsort(-pred)[0:5])))
        except Exception as e:
            print("Can't find " + bin_path)
    print('imag_count %d, top1_accuracy %.3f top5_accuracy %.3f'%(image_cnt,top1_cnt/image_cnt,top5_cnt/image_cnt))
