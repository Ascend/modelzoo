import os
import threading
import queue
import imageio


class ImageWriter:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            ImageWriter.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, max_num_threads=1, max_queue_size=64, timeout=100):
        self.queue = queue.Queue(max_queue_size)
        self.threads_pool = []
        self.sentinel = (None, None)
        self.timeout = timeout
        self.max_num_threads = max_num_threads

    def worker(self):
        while True:
            try:
                elem = self.queue.get(True)
                if id(elem) == id(self.sentinel):
                    self.end()
                    break
                target_path, im_data = elem
                imageio.imwrite(target_path, im_data)
                # print(f'Write to {target_path}')
            except:
                pass

    def __del__(self):
        for t in self.threads_pool:
            try:
                t.join()
            except:
                pass
        print('Processing remaining elements')

        while True:
            try:
                elem = self.queue.get(False)
                assert id(elem) == id(self.sentinel), '[Warning] Remain elements in writing queue'
            except queue.Empty:
                break

    def put_to_queue(self, target_path, im_data):
        self.queue.put((target_path, im_data))
        if len(self.threads_pool) <= self.max_num_threads:
            t = threading.Thread(target=self.worker, args=())
            t.start()
            self.threads_pool.append(t)

    def end(self):
        self.queue.put(self.sentinel)
