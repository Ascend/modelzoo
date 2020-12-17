#!/usr/bin/env python
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
import time
from subprocess import Popen

import pyarrow
import redis

from xt.framework.register import Registers


@Registers.comm.register
class ShareByRedis(object):
    def __init__(self, comm_info):
        """ init redis component """

        super(ShareByRedis, self).__init__()
        # For master, there is no 'addr' parameter given.
        self.ip_addr = comm_info.get("addr", "127.0.0.1")
        self.port = comm_info.get("port", 6379)
        self.password = comm_info.get("password", None)
        self.strat_redis = False
        if self.ip_addr == "127.0.0.1":
            self.start()

        self.redis = redis.Redis(host=self.ip_addr, port=self.port, db=0)

    def send(self, data, name=None, block=True):
        """ send data to redis server """

        data_buffer = pyarrow.serialize(data).to_buffer()
        self.redis.set(name, data_buffer)

    def recv(self, name=None):
        """ recieve data from redis server """

        data_buffer = self.redis.get(name)
        data = pyarrow.deserialize(data_buffer)
        return data

    def delete(self, name):
        """ delete items in redis server """

        self.redis.delete(name)

    def start(self):
        """ start redis server """
        try:
            redis.Redis(host=self.ip_addr, port=self.port, db=0).ping()
        except redis.ConnectionError:
            Popen("echo save '' | setsid redis-server -", shell=True)
            self.strat_redis = True
            time.sleep(0.1)

    def close(self):
        """ shutdown redis client """
        # self.redis.flushdb()
        self.redis.shutdown(nosave=True)
        print("redis shutdown")
