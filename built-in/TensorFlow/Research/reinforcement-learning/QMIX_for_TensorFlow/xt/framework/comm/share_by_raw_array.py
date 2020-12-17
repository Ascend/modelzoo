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

from ctypes import addressof, c_ubyte, memmove
from multiprocessing import Queue, RawArray
import pyarrow

import lz4.frame

from xt.framework.register import Registers


@Registers.comm.register
class ShareByRawArray(object):
    def __init__(self, comm_info):
        """ init share memory """
        super(ShareByRawArray, self).__init__()

        self.size_shared_mem = comm_info.get("size", 100000000)
        self.agent_num = comm_info.get("agent_num", 4)

        self.control_q = Queue()
        self.mem = RawArray(c_ubyte, self.size_shared_mem)
        self.size_mem_agent = int(self.size_shared_mem / self.agent_num)

    def send(self, data, name=None, block=True):
        """ put data in share memory """
        data_id, data = data
        msg = lz4.frame.compress(pyarrow.serialize(data).to_buffer())

        memmove(addressof(self.mem) + int(data_id * self.size_mem_agent), msg, len(msg))

        self.control_q.put((data_id, len(msg)))

    def recv(self, name=None):
        """ get data from share memory """
        data_id, len_data = self.control_q.get()

        data = pyarrow.deserialize(
            lz4.frame.decompress(
                memoryview(self.mem)[
                    int(data_id * self.size_mem_agent) : int(
                        data_id * self.size_mem_agent + len_data
                    )
                ]
            )
        )

        return data

    def recv_bytes(self):
        """ get data from share memory without deserialize """
        data_id, len_data = self.control_q.get()

        return memoryview(self.mem)[
            int(data_id * self.size_mem_agent) : int(
                data_id * self.size_mem_agent + len_data
            )
        ]

    def send_bytes(self, data):
        """ put data in share memory without serialize """
        data_id, data_buffer = data
        memmove(
            addressof(self.mem) + int(data_id) * self.size_mem_agent,
            data_buffer,
            len(data_buffer),
        )

        self.control_q.put((data_id, len(data_buffer)))

    def send_multipart(self, data):
        """ put multi-data in share memory without serialize """
        data_id, data_buffer = data
        self.control_q.put(len(data_buffer))
        for _id, _buffer in zip(data_id, data_buffer):
            memmove(
                addressof(self.mem) + int(_id) * self.size_mem_agent,
                _buffer,
                len(_buffer),
            )
            self.control_q.put((_id, len(_buffer)))

    def recv_multipart(self):
        """ get multi-data from share memory without deserialize """
        len_data = self.control_q.get()
        data_id = []
        data_buffer = []
        for _ in range(len_data):
            _id, len_buff = self.control_q.get()
            data_id.append(_id)
            data_buffer.append(
                memoryview(self.mem)[
                    int(_id * self.size_mem_agent) : int(
                        _id * self.size_mem_agent + len_buff
                    )
                ]
            )

        return data_buffer

    def close(self):
        """ close """
        pass
