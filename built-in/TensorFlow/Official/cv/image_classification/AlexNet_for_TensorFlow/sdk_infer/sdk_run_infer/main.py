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

from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi
import os
import json
import sys
import time

cur_path = os.path.dirname(os.path.realpath(__file__))
res_dir = 'result'
res_acc_dir = 'result_acc'


def scan_img_files(file_path):
    """scan image files ending with jpg and jpeg under filepath"""
    files = os.listdir(file_path)
    return [f_name for f_name in files
            if f_name.lower().endswith('.jpg') or f_name.lower().endswith('.jpeg')]


def write_infer_res(result_dir, img_name, infer_result_str):
    """write inference result under result_dir"""
    json_res = json.loads(infer_result_str)
    res_vec = json_res.get('MxpiClass')
    f_name = os.path.join(cur_path, result_dir, img_name[:-5] + "_1.txt")
    with open(f_name, 'w') as f_res:
        cls_ids = [str(item.get('classId')) + " " for item in res_vec]
        f_res.writelines(cls_ids)
        f_res.write('\n')


if __name__ == '__main__':
    start_time = time.time()
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    pipeline_file = "../pipeline/alexnet_tf.pipeline"
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

        # Inputs data to a specified stream based on stream_name.
        stream_name = b'im_alexnet'
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
