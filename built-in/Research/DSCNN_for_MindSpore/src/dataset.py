import os
import numpy as np
import mindspore.dataset as de


class npyDataset(object):
    def __init__(self, data_dir, data_type, h, w):
        super(npyDataset, self).__init__()
        self.data = np.load(os.path.join(data_dir, '{}_data.npy'.format(data_type)))
        self.data = np.reshape(self.data, (-1, 1, h, w))
        self.label = np.load(os.path.join(data_dir, '{}_label.npy'.format(data_type)))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        # return data, label
        return data.astype(np.float32), label.astype(np.int32)


def audio_dataset(data_dir, data_type, h, w, batch_size):
    if 'testing' in data_dir:
        shuffle = False
    else:
        shuffle = True
    dataset = npyDataset(data_dir, data_type, h, w)
    de_dataset = de.GeneratorDataset(dataset, ["feats", "labels"], shuffle=shuffle)
    de_dataset = de_dataset.batch(batch_size, drop_remainder=False)
    return de_dataset
