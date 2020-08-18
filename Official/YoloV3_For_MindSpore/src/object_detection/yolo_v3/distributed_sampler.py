from __future__ import division
import math
import numpy as np

'''
Source: refer reid DistributedPKSampler and modify a little
'''

class DistributedSampler():
    def __init__(self, dataset_size, num_replicas=None, rank=None, shuffle=True):
        # dataset_size: num images in dataset
        # num_replicas: device_num

        if num_replicas is None:
            print("***********Setting world_size to 1 since it is not passed in ******************")
            num_replicas = 1
        if rank is None:
            ##TODO, get rank if not set
            print("***********Setting rank to 0 since it is not passed in ******************")
            rank = 0
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # total_size: drop image which can't be divided by device, left is total image size used to trained
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            # np.array type. number from 0 to len(dataset_size)-1, used as index of dataset
            indices = indices.tolist()
            self.epoch += 1
            # change to list type
        else:
            indices = list(range(self.dataset_size))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    # def set_epoch(self, epoch):
    #     self.epoch = epoch

