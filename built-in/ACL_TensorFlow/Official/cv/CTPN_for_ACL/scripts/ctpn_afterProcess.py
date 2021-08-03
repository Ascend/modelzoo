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
import os
import sys
import shutil
import time
import numpy as np
from PIL import Image
import PIL
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
import cv2
import argparse

parser = argparse.ArgumentParser(description="ctpn afterprocess.")

parser.add_argument("--result_path",type=str,default="../results/ctpn",help="The path of the result_Files.")
parser.add_argument("--after_result",type=str,default="../output_temp",help="The path of the txt file")
parser.add_argument("--img_conf_path",type=str,default="./img_info",help="The path of img config path")

args = parser.parse_args()

def get_img_config():
    img_conf = args.img_conf_path
    with open(img_conf, 'r')as f:
        img_info_list = f.read().split('\n')[:-1]
    img_info_dict = {}
    for i in img_info_list:
        tmp_list = i.split(' ')
        idx = int(tmp_list[0])
        imgName = tmp_list[1]
        height = int(tmp_list[2])
        width = int(tmp_list[3])
        channel = int(tmp_list[4])
        rh = float(tmp_list[5])
        rw = float(tmp_list[6])
        img_info_dict[idx] = {
                'imgName' : imgName,
                'height' : height,
                'width' : width,
                'channel' : channel,
                'rh' : rh,
                'rw' : rw
                }
    return img_info_dict

def main(result_path, res_out, img_info_dict):
    if os.path.exists(res_out):
        shutil.rmtree(res_out)
    os.makedirs(res_out)
    test_len = len(img_info_dict.keys())
    for test_idx in range(test_len):
        img_name = img_info_dict[test_idx]['imgName']
        img_temp = str(img_name)
        h = img_info_dict[test_idx]['height']
        w = img_info_dict[test_idx]['width']
        c = img_info_dict[test_idx]['channel']
        bbox_pred_val = np.fromfile(result_path + "/davinci_" + img_temp + "_output0.bin", dtype=np.float32)
        bbox_pred_val = bbox_pred_val.reshape(1,38,67,40)
        cls_prob_val = np.fromfile(result_path + "/davinci_" + img_temp + "_output1.bin", dtype=np.float32)
        cls_prob_val = cls_prob_val.reshape(1,38,670,2)
        im_info = np.array([h, w, c]).reshape([1, 3])
        textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]
        img_shape = (h, w, c)
        textdetector = TextDetector(DETECT_MODE='H')
        boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img_shape[:2])
        boxes = np.array(boxes, dtype=np.int)
        rh = img_info_dict[test_idx]['rh']
        rw = img_info_dict[test_idx]['rw']
        with open(res_out + "/res_" + img_temp + ".txt", "w") as f:
            for i, box in enumerate(boxes):
                box[0] = box[0]/rw
                box[1] = box[1]/rh
                box[4] = box[4]/rw
                box[5] = box[5]/rh
                line = ",".join(str(box[k]) for k in [0,1,4,5])
                line = line + '\n'
                f.writelines(line)

if __name__ == '__main__':
    result_path = args.result_path
    res_out = args.after_result
    img_info_dict = get_img_config()
    main(result_path, res_out, img_info_dict)
