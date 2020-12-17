# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

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
# ============================================================================
#!/usr/bin/env python

import os
import time
from multiprocessing import Queue
from subprocess import PIPE, Popen

import lz4.frame
# from pyarrow import deserialize, plasma, serialize

from xt.framework.register import Registers
import pickle

@Registers.comm.register
class ShareByPlasma(object):
    def __init__(self, comm_info):
        """ init plasma component """
        super(ShareByPlasma, self).__init__()
        self.size_shared_mem = comm_info.get("size", 1000000000)
        self.path = comm_info.get("path", "/tmp/plasma" + str(os.getpid()))
        self.compress = comm_info.get("compress", True)

        self.control_q = Queue()
        self.client = {}
        self.start()

    def send(self, data, name=None, block=True):
        """ send data to plasma server """
        # client = self.connect()

        # data_buffer = lz4.frame.compress(serialize(data).to_buffer())
        # object_id = client.put_raw_buffer(data_buffer)
        pickled_data = pickle.dumps(data)
        data_buffer = lz4.frame.compress(pickled_data)
        self.control_q.put(data_buffer)

        # del data
        if data["ctr_info"].get("cmd") == "train":
            keys = []
            for key in data["data"].keys():
                keys.append(key)
            for key in keys:
                del data["data"][key]
        elif data["ctr_info"].get("cmd") == "predict":
            del data["data"]


    def recv(self, name=None, block=True):
        """ receive data from plasma server """
        object_id = self.control_q.get()
        client = self.connect()
        data = deserialize(lz4.frame.decompress(client.get_buffers([object_id])))
        client.delete([object_id])

        return data

    def send_bytes(self, data_buffer):
        """ send data to plasma server without serialize """
        client = self.connect()
        object_id = client.put_raw_buffer(data_buffer)
        self.control_q.put(object_id)

    def recv_bytes(self):
        """ receive data from plasma server without deserialize """
        object_id = self.control_q.get()
        # client = self.connect()
        # data_buffer = client.get_buffers([object_id])
        # client.delete([object_id])

        return object_id, 0

    def delete(self, object_id):
        # client = self.connect()
        # client.delete([object_id])
        pass

    def send_multipart(self, data_buffer):
        """ send multi-data to plasma server without serialize """
        client = self.connect()
        self.control_q.put(len(data_buffer))
        for _buffer in data_buffer:
            objec_id = client.put_raw_buffer(_buffer)
            self.control_q.put(objec_id)

    def recv_multipart(self):
        """ recieve multi-data from plasma server without deserialize """
        len_data = self.control_q.get()
        object_id = []
        client = self.connect()
        for _ in range(len_data):
            _object_id = self.control_q.get()
            object_id.append(_object_id)

        data_buffer = client.get_buffers(object_id)
        client.delete(object_id)

        return data_buffer

    def start(self):
        """ start plasma server """
        try:
            client = plasma.connect(self.path, int_num_retries=2)
        except:
            Popen(
                "plasma_store -m {} -s {}".format(self.size_shared_mem, self.path),
                shell=True,
                stderr=PIPE,
            )
            print(
                "plasma_store -m {} -s {} is acitvated!".format(
                    self.size_shared_mem, self.path
                )
            )
            time.sleep(0.1)

    def connect(self):
        """ connect to plasma server """
        pid = os.getpid()
        if pid in self.client:
            return self.client[pid]
        else:
            self.client[pid] = plasma.connect(self.path)
            return self.client[pid]

    def close(self):
        """ close plasma server """
        os.system("pkill -9 plasma")
