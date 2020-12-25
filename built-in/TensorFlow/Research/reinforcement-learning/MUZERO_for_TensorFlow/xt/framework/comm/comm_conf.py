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
import socket
import time
from subprocess import Popen

import redis

MAX_ACTOR_NUM = 40
MAX_LEARNER_NUM = 10
START_PORT = 20000
PORTNUM_PERLEARNER = MAX_ACTOR_NUM + 1


class CommConf(object):
    def __init__(self):
        try:
            redis.Redis(host="127.0.0.1", port=6379, db=0).ping()
        except redis.ConnectionError:
            Popen("echo save '' | setsid redis-server -", shell=True)
            time.sleep(0.3)

        self.redis = redis.Redis(host="127.0.0.1", port=6379, db=0)
        self.pool_name = "port_pool"
        if not self.redis.exists(self.pool_name):
            self.init_portpool()

    def init_portpool(self):
        ''' init port pool '''
        start_port = START_PORT
        try_num = 10

        for _ in range(MAX_LEARNER_NUM):
            for _ in range(try_num):
                check_flag, next_port = self.check_learner_port(start_port)
                if not check_flag:
                    break
                else:
                    start_port = next_port

            self.redis.lpush(self.pool_name, start_port)
            self.redis.incr('port_num', amount=1)
            self.redis.incr('max_port_num', amount=1)

            start_port = next_port

    def get_start_port(self):
        ''' get start port '''
        if int(self.redis.get('port_num')) == 0:
            raise Exception("Dont have available port")

        start_port = self.redis.lpop(self.pool_name)
        self.redis.decr('port_num', amount=1)
        return int(start_port)

    def release_start_port(self, start_port):
        ''' release start port '''
        self.redis.lpush(self.pool_name, start_port)
        self.redis.incr('port_num', amount=1)

        if self.redis.get('port_num') == self.redis.get('max_port_num'):
            self.redis.delete('port_num')
            self.redis.delete('max_port_num')
            self.redis.delete('port_pool')
            print("shutdown redis")
            self.redis.shutdown(nosave=True)

        return

    def check_learner_port(self, start_port):
        ''' check if multi-port is in use '''
        ip = "localhost"
        for i in range(PORTNUM_PERLEARNER):
            if self.check_port(ip, start_port + i):
                return True, start_port + i + 1
        return False, start_port + PORTNUM_PERLEARNER

    def check_port(self, ip, port):
        ''' check if port  is in use '''
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, int(port)))
            s.shutdown(2)
            print("port is used", int(port))
            return True
        except BaseException:
            return False


def get_port(start_port):
    ''' get port used by module '''
    predict_port = start_port + 1
    if (predict_port + MAX_ACTOR_NUM - start_port) > PORTNUM_PERLEARNER:
        raise Exception("port num is not enough")

    return start_port, predict_port


def test():
    ''' test interface'''
    test_comm_conf = CommConf()
    redis_key = 'port_pool'
    print("{} len: {}".format(redis_key, test_comm_conf.redis.llen(redis_key)))
    for _ in range(test_comm_conf.redis.llen(redis_key)):
        pop_val = test_comm_conf.redis.lpop(redis_key)
        print("pop val: {} from '{}'".format(pop_val, redis_key))
    start = time.time()

    test_comm_conf.init_portpool()
    print("use time", time.time() - start)

    train_port = get_port(20000)
    print(train_port)
    # test_comm_conf.release_start_port(train_port[0])


if __name__ == "__main__":
    test()
