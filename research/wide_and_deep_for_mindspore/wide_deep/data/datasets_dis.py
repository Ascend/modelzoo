import os
import numpy as np
import math
import mindspore.dataset.engine as de

import pandas as pd


class H5Dataset():
    max_length = 39

    def __init__(self, data_path, train_mode=True, train_num_of_parts=21, test_num_of_parts=3):
        self._hdf_data_dir = data_path
        self._is_training = train_mode

        if self._is_training:
            self._file_prefix = 'train'
            self._num_of_parts = train_num_of_parts
        else:
            self._file_prefix = 'test'
            self._num_of_parts = test_num_of_parts

        self.data_size = self._bin_count(self._hdf_data_dir, self._file_prefix,
                                         self._num_of_parts)
        print("data_size: {}".format(self.data_size))

    def _bin_count(self, hdf_data_dir, file_prefix, num_of_parts):
        size = 0
        for part in range(num_of_parts):
            _y = pd.read_hdf(os.path.join(hdf_data_dir,
                                          file_prefix + '_output_part_' + str(
                                              part) + '.h5'))
            size += _y.shape[0]
        return size

    def _iterate_hdf_files_(self, num_of_parts=None,
                            shuffle_block=False):
        """
        iterate among hdf files(blocks). when the whole data set is finished, the iterator restarts
            from the beginning, thus the data stream will never stop
        :param train_mode: True or false,false is eval_mode,
            this file iterator will go through the train set
        :param num_of_parts: number of files
        :param shuffle_block: shuffle block files at every round
        :return: input_hdf_file_name, output_hdf_file_name, finish_flag
        """
        parts = np.arange(num_of_parts)
        while True:
            if shuffle_block:
                for _ in range(int(shuffle_block)):
                    np.random.shuffle(parts)
            for i, p in enumerate(parts):
                yield os.path.join(self._hdf_data_dir,
                                   self._file_prefix + '_input_part_' + str(
                                       p) + '.h5'), \
                      os.path.join(self._hdf_data_dir,
                                   self._file_prefix + '_output_part_' + str(
                                       p) + '.h5'), i + 1 == len(parts)

    def _generator(self, X, y, batch_size, shuffle=True):
        """
        should be accessed only in private
        :param X:
        :param y:
        :param batch_size:
        :param shuffle:
        :return:
        """
        number_of_batches = np.ceil(1. * X.shape[0] / batch_size)
        counter = 0
        finished = False
        sample_index = np.arange(X.shape[0])
        if shuffle:
            for _ in range(int(shuffle)):
                np.random.shuffle(sample_index)
        assert X.shape[0] > 0
        while True:
            batch_index = sample_index[
                          batch_size * counter: batch_size * (counter + 1)]
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch, finished
            if counter == number_of_batches:
                counter = 0
                finished = True

    def batch_generator(self, batch_size=1000,
                        random_sample=False, shuffle_block=False):
        """
        :param train_mode: True or false,false is eval_mode,
        :param batch_size
        :param num_of_parts: number of files
        :param random_sample: if True, will shuffle
        :param shuffle_block: shuffle file blocks at every round
        :return:
        """
        dataset = []
        for hdf_in, hdf_out, _ in self._iterate_hdf_files_(self._num_of_parts,
                                                           shuffle_block):
            start = stop = None
            X_all = pd.read_hdf(hdf_in, start=start, stop=stop).values
            y_all = pd.read_hdf(hdf_out, start=start, stop=stop).values
            data_gen = self._generator(X_all, y_all, batch_size,
                                       shuffle=random_sample)
            finished = False

            while not finished:
                X, y, finished = data_gen.__next__()
                X_id = X[:, 0:self.max_length]
                X_va = X[:, self.max_length:]
                element = [np.array(X_id.astype(dtype=np.int32)), np.array(
                    X_va.astype(dtype=np.float32)), np.array(
                    y.astype(dtype=np.float32))]
                dataset.append(element)

class DistributedSampler():
    def __init__(self, data_size, local_rank, rank_size):
        self.__num_rows = data_size
        self.__local_rank = local_rank
        self.__rank_size = rank_size
        self.samples_per_rank = int(math.ceil(self.__num_rows / float(self.__rank_size)))
        self.total_num_samples = samples_per_rank * self.__rank_size

    def __iter__(self):
        indices = list(range(self.__num_rows))
        indices.extend(indices[:self.total_num_samples-len(indices)])
        indices = indices[self.__local_rank:self.total_num_samples:self.__rank_size]
        return iter(indices)

def create_dataset(dir, train_mode=True, epochs=1, batch_size=1, local_rank=None, rank_size=None):
    data_para = {
        'batch_size': batch_size,
    }
    if train_mode:
        data_para['random_sample'] = True
        data_para['shuffle_block'] = True

    h5_dataset = H5Dataset(data_path=dir, train_mode=train_mode)
    train_eval_gen = h5_dataset.batch_generator(**data_para)
    if local_rank and rank_size:
        sampler = DistributedSampler(h5_dataset.data_size, local_rank=local_rank, rank_size=rank_size)
    ds = de.GeneratorDataset(train_eval_gen, ["ids", "weights", "labels"], sampler=sampler)
    numbers_of_batch = math.ceil(h5_dataset.data_size / batch_size)
    ds.set_dataset_size(numbers_of_batch)
    ds = ds.repeat(epochs)
    print("----numbers_of_batch----", numbers_of_batch)
    return ds
