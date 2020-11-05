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
