import math
import numpy as np


class DistributedSampler(object):
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset_length)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples
