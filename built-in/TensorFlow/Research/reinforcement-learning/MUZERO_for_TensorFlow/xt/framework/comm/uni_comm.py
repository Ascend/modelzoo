# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading

from xt.framework.register import Registers


class UniComm(object):
    def __init__(self, comm_name, **comm_info):
        super(UniComm, self).__init__()
        self.comm = Registers.comm[comm_name](comm_info)
        self.lock = threading.Lock()

    def send(self, data, name=None, block=True):
        """ common send interface """
        return self.comm.send(data, name, block)

    def recv(self, name=None, block=True):
        """ common recieve interface """
        return self.comm.recv(name, block)

    def send_bytes(self, data):
        """ common send_bytes interface """
        return self.comm.send_bytes(data)

    def recv_bytes(self):
        """ common recv_bytes interface """
        return self.comm.recv_bytes()

    def send_multipart(self, data):
        """ common send_multipart interface """
        return self.comm.send_multipart(data)

    def recv_multipart(self):
        """ common recv_multipart interface """
        return self.comm.recv_multipart()

    def delete(self, name):
        return self.comm.delete(name)

    def close(self):
        print("start_close_comm")
        with self.lock:
            try:
                self.comm.close()
            except AttributeError as ex:
                print("please complete your comm close function")
