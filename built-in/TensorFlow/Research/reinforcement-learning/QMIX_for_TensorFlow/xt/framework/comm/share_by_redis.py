#!/usr/bin/env python
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
