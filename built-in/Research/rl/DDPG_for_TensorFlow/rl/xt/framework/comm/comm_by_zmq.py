# import pyarrow
import pickle
import zmq
from xt.framework.register import Registers


@Registers.comm.register
class CommByZmq(object):
    def __init__(self, comm_info):
        super(CommByZmq, self).__init__()
        # For master, there is no 'addr' parameter given.
        addr = comm_info.get("addr", "*")
        port = comm_info.get("port")
        zmq_type = comm_info.get("type", "PUB")

        comm_type = {
            "PUB": zmq.PUB,
            "SUB": zmq.SUB,
            "PUSH": zmq.PUSH,
            "PULL": zmq.PULL,
            "REP": zmq.REP,
            "REQ": zmq.REQ,
        }.get(zmq_type)

        context = zmq.Context()
        socket = context.socket(comm_type)

        if "*" in addr:
            socket.bind("tcp://*:" + str(port))
        else:
            socket.connect("tcp://" + str(addr) + ":" + str(port))

        self.socket = socket

    def send(self, data, name=None, block=True):
        # msg = pyarrow.serialize(data).to_buffer()
        msg = pickle.dumps(data)
        self.socket.send(msg)

    def recv(self, name=None, block=True):
        msg = self.socket.recv()
        # data = pyarrow.deserialize(msg)
        data = pickle.loads(msg)
        return data

    def send_bytes(self, data):
        self.socket.send(data, copy=False)

    def recv_bytes(self):
        data = self.socket.recv()
        return data

    def close(self):
        if self.socket:
            self.socket.close()
