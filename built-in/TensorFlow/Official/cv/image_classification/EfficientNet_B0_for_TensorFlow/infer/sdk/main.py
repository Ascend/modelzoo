#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from StreamManagerApi import RoiBoxVector
from StreamManagerApi import RoiBox
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput
import os
import json
import sys
import time
import cv2
import numpy as np

cur_path = os.path.dirname(os.path.realpath(__file__))
res_dir = 'result'
res_acc_dir = 'result_acc'


def scan_img_files(file_path):
    """scan jpg and jpeg image file under file_path"""
    files = os.listdir(file_path)
    return [f_name for f_name in files
            if f_name.lower().endswith('.jpg') or f_name.lower().endswith('.jpeg')]


def write_infer_res(result_dir, img_name, infer_result_str):
    """write inference result"""
    json_res = json.loads(infer_result_str)
    res_vec = json_res.get('MxpiClass')
    f_name = os.path.join(cur_path, result_dir, img_name[:-5] + "_1.txt")
    with open(f_name, 'w') as f_res:
        cls_ids = [str(item.get('classId')) + " " for item in res_vec]
        f_res.writelines(cls_ids)
        f_res.write('\n')


def calc_roi_vec(image_file_path):
    """calc roi vector of image"""
    raw_img = cv2.imread(image_file_path)
    img_height, img_width, _ = np.shape(raw_img)
    target_size = int(min(img_height, img_width) * 0.874)
    amount_crop_width = img_width - target_size
    amount_crop_height = img_height - target_size
    left_half = amount_crop_width // 2
    top_half = amount_crop_height // 2
    left = left_half if left_half % 2 == 0 else left_half + 1
    top = top_half if top_half % 2 == 0 else top_half + 1
    right = left + target_size - 1 if (left + target_size - 1) % 2 == 1 else left + target_size - 2
    bottom = top + target_size - 1 if (top + target_size - 1) % 2 == 1 else top + target_size - 2
    roi_box_vec = RoiBoxVector()
    roi = RoiBox()
    roi.x0 = left
    roi.y0 = top
    roi.x1 = right
    roi.y1 = bottom
    roi_box_vec.push_back(roi)
    return roi_box_vec


if __name__ == '__main__':
    start_time = time.time()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_file = "../pipeline/efficientnetb0_tf.pipeline"
    with open(pipeline_file, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()
    if len(sys.argv) == 2:
        img_path = sys.argv[1]
    else:
        img_path = "./"

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(res_acc_dir):
        os.makedirs(res_acc_dir)

    img_files = scan_img_files(img_path)
    for idx, file_name in enumerate(img_files):
        img_file_path = os.path.join(img_path, file_name)
        with open(img_file_path, 'rb') as f:
            data_input.data = f.read()

        roi_vec = calc_roi_vec(img_file_path)
        if not roi_vec:
            print("roi vector is None")
            continue
        data_input.roiBoxs = roi_vec

        # Inputs data to a specified stream based on stream_name.
        stream_name = b'im_efficientnetb0'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendDataWithUniqueId(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying stream_name and unique_id.
        infer_result = stream_manager_api.GetResultWithUniqueId(stream_name, unique_id, 3000)
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            print("file path: {}, file name: {}".format(img_path, file_name))
            exit()
        # print the infer result
        infer_res = infer_result.data.decode()
        print("process index: {}, img: {}, infer result: {}".format(idx, file_name, infer_res))
        write_infer_res(res_dir, file_name, infer_res)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
    end_time = time.time()
    print("total cost {} seconds.".format(end_time - start_time))
