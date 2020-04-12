import numpy as np
import bisect


class ConcatDataset(object):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """
    @staticmethod
    def countsum(sequence):
        reslist = []
        sumlen = 0
        for data in sequence:
            datalen = len(data)
            sumlen += datalen            # t = t + l
            reslist += [sumlen]          # reslist.append(l + t)
        return reslist

    def __init__(self, datasets):
        # super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.accumulate_sizes = self.countsum(self.datasets)
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

    def __len__(self):
        return self.accumulate_sizes[-1]

    # TODO: this copy: import ConcatDataset from torch.utils.data.dataset, need modify
    def __getitem__(self, index):   # idx --> index; dataset_idx --> dIndex; sample_idx = sIndex
        if index < 0:
            if abs(index) > len(self):
                raise ValueError("index exceeded dataset length")
            index += len(self)   # idx + = len(self)

        # dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if index < self.accumulate_sizes[0]:
            dIndex = 0
            sIndex = index
        else:
            dIndex = bisect.bisect_right(self.accumulate_sizes, index)
            sIndex = index - self.accumulate_sizes[dIndex - 1]
        return self.datasets[dIndex][sIndex]
