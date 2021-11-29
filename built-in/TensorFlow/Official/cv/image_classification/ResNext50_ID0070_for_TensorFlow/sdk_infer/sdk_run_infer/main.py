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

# import StreamManagerApi.py
from StreamManagerApi import *
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import datetime


def crop_image(image_path):
    roiVector = RoiBoxVector()
    roi = RoiBox()
    img = cv2.imread(image_path)
    central_fraction = 0.8
    height, width, _ = np.shape(img)
    target_h = int(height * central_fraction) + 1
    target_w = int(width * central_fraction) + 1
    amount_to_be_cropped_h = (height - target_h)
    amount_to_be_cropped_w = (width - target_w)
    crop_y = amount_to_be_cropped_h // 2
    crop_x = amount_to_be_cropped_w // 2
    print('crop image, width:{}, height:{}.'.format(width, height))
    print('crop image, x0:{}, y0:{}, x1:{}, y1:{}.'.format(crop_x, crop_y,
                                                           crop_x + target_w,
                                                           crop_y + target_h))
    roi.x0 = crop_x
    roi.y0 = crop_y
    roi.x1 = crop_x + target_w
    roi.y1 = crop_y + target_h
    roiVector.push_back(roi)
    return roiVector


def save_infer_result(infer_result):
    load_dict = json.loads(infer_result)
    if load_dict.get('MxpiClass') is None:
        with open(res_dir_name + "/" + file_name[:-5] + '.txt', 'w') as f_write:
            f_write.write("")
    else:
        res_vec = load_dict['MxpiClass']
        with open(res_dir_name + "/" + file_name[:-5] + '_1.txt', 'w') as f_write:
            list1 = [str(item.get("classId") - 1) + " " for item in res_vec]
            f_write.writelines(list1)
            f_write.write('\n')


if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/resnext50_opencv.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()

    dir_name = './val_union/'
    res_dir_name = 'result'
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        print(file_name)
        file_path = dir_name + file_name
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            dataInput.data = f.read()

        dataInput.roiBoxs = crop_image(file_path)
        # Inputs data to a specified stream based on streamName.
        streamName = b'im_resnext50'
        inPluginId = 0
        uniqueId = streamManagerApi.SendData(streamName, inPluginId, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResult(streamName, uniqueId)
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            exit()
        # print the infer result
        print(inferResult.data.decode())
        save_infer_result(inferResult.data.decode())


    # destroy streams
    streamManagerApi.DestroyAllStreams()
