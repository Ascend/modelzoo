# -*- coding:utf-8 -*-
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
from StreamManagerApi import *
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import datetime

if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./pipeline/ctpn_single_cv.pipeline", 'rb') as f:
        pipeline_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    dir_name = 'Challenge2_Test_Task12_Images/'
    res_dir_name = 'icdar2013_ctpn_ms_opencv'
    file_list = os.listdir(dir_name)
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)
    for file_name in file_list:
        start_time = datetime.datetime.now()
        print(file_name)
        file_path = os.path.join(dir_name, file_name)
        if file_name.endswith(".JPG") or file_name.endswith(".jpg"):
            with open(file_path, 'rb') as f:
                data_input.data = f.read()

        # Inputs data to a specified stream based on streamName.
        stream_name = b'classification+detection'
        in_plugin_id = 0
        unique_id = stream_manager_api.SendDataWithUniqueId(
            stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId
        infer_result = stream_manager_api.GetResultWithUniqueId(
            stream_name, unique_id, 3000)
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()

        # print the infer result
        print(infer_result.data.decode())

        load_dict = json.loads(infer_result.data.decode())
        if load_dict.get('MxpiTextObject') is None:
            with open(res_dir_name + "/" + 'res_' + file_name[:-4] +
                      '.txt', 'w') as f_write:
                f_write.write("")
            continue
        res_vec = load_dict['MxpiTextObject']
        boxes = []
        for res in res_vec:
            boxes.append([int(res['x0']), int(res['y0']), int(res['x2']),
                          int(res['y2'])])
        output_file = res_dir_name + "/" + 'res_' + file_name
        boxes = np.array(boxes, dtype=float)
        im = cv2.imread(file_path)
        for i, box in enumerate(boxes):
            cv2.rectangle(im, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)
        cv2.imwrite(output_file, im)
        with open(res_dir_name + "/" + 'res_' + file_name[:-4] + '.txt', 'w') \
                as f_write:
            for i, box in enumerate(boxes):
                line = ",".join(str(int(box[k])) for k in range(4))
                f_write.writelines(line)
                f_write.write('\n')
        end_time = datetime.datetime.now()
        print('CTPN sdk run time: {}'.format(
            (end_time - start_time).microseconds))

    # destroy streams
    stream_manager_api.DestroyAllStreams()
