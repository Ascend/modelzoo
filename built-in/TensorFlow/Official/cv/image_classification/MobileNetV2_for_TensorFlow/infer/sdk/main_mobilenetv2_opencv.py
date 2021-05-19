# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

from StreamManagerApi import *
import os
import cv2
import json
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
import datetime

IMAGE_DIR_NAME = './val_images/'
RES_DIR_NAME = 'mobilenetv2_npu_result'
PIPELINE_PATH = 'pipeline/mobilenetv2_opencv.pipeline'


def do_infer_and_get_result(stream_api, data_input, image_dir_name, file_lists):
    for file_name in file_lists:
        print(file_name)
        file_path = os.path.join(image_dir_name, file_name)
        if file_name.endswith(".JPG") or file_name.endswith(".jpg") or \
                file_name.endswith(".JPEG") or file_name.endswith(".jpeg"):
            with open(file_path, 'rb') as fs:
                data_input.data = fs.read()

        # Inputs data to a specified stream based on stream_name.
        stream_name = b'mobilenetv2'
        in_plugin_id = 0
        unique_id = stream_api.SendData(
            stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying stream_name and unique_id
        start_time = datetime.datetime.now()
        infer_result = stream_api.GetResult(stream_name, unique_id)
        end_time = datetime.datetime.now()
        print('sdk run time: {}'.format((end_time - start_time).microseconds))
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()

        # print the infer result
        print(infer_result.data.decode())

        load_dict = json.loads(infer_result.data.decode())
        file_suffix = file_name[:-5] + '.txt'
        full_file_name = os.path.join(RES_DIR_NAME, file_suffix)
        if load_dict.get('MxpiClass') is None:
            with open(full_file_name, 'w') \
                    as f_write:
                f_write.write("")
            continue
        resVec = load_dict['MxpiClass']

        res_suffix = file_name[:-5] + '_1.txt'
        res_file_name = os.path.join(RES_DIR_NAME, res_suffix)
        with open(res_file_name, 'w') \
                as f_write:
            list1 = [str(item.get("classId")) + " " for item in resVec]
            f_write.writelines(list1)
            f_write.write('\n')


if __name__ == '__main__':
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(PIPELINE_PATH, 'rb') as f:
        pipe_line_str = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipe_line_str)

    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_inputs = MxDataInput()

    file_list = os.listdir(IMAGE_DIR_NAME)
    if not os.path.exists(RES_DIR_NAME):
        os.makedirs(RES_DIR_NAME)
    do_infer_and_get_result(stream_manager_api, data_inputs,
                            IMAGE_DIR_NAME, file_list)

    # destroy streams
    stream_manager_api.DestroyAllStreams()
