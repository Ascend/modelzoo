import os 
import threading
import queue 


class PrefetchGenerator():
    def __init__(self, generator, num_threads=1, max_queue_size=64):
        self.queue = queue.Queue(max_queue_size)
        # Put a first element into the queue, and initialize our thread
        self.generator = generator
        self.threads_pool = []
        for i in range(num_threads):
            t = threading.Thread(target=self.worker, args=(), daemon=True)
            t.start()
            self.threads_pool.append(t)
    
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            PrefetchGenerator.__instance = super().__new__(cls)
        return cls.__instance

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item
    
    def worker(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __del__(self):
        self.stop()

    def stop(self):
        for t in self.threads_pool:
            t.join()
        while True: # Flush the queue
            try:
                self.queue.get(False)
            except Queue.Empty:
                break

        
        # self.t.join()
