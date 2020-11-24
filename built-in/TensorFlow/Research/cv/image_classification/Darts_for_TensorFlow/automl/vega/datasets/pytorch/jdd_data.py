# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for JDD dataset."""
import os
import os.path
import numpy as np
import torch
from .common.dataset import Dataset
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps
from vega.datasets.conf.jdd import JDDConfig


@ClassFactory.register(ClassType.DATASET)
class JDDData(Dataset):
    """Construct the class of JDDData dataset, which is a subclass of Dateset.

    :param train: if the mdoe is train or false, defaults to True
    :type train: bool, optional
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = JDDConfig()

    def __init__(self, **kwargs):
        """Construct the DistributorBaseClass class.."""
        super(JDDData, self).__init__(**kwargs)
        self.dataset_init()

    def dataset_init(self):
        """Costruct method, which will load some dateset information."""
        self.args.root = FileOps.download_dataset(self.args.root)
        self.train_list = os.listdir(self.args.root)
        self.train_list.sort()
        self.nc = len(self.train_list)

    def __len__(self):
        """Get the length of the dataset.

        :return: the length of the dataset
        :rtype: int
        """
        return self.nc

    def __getitem__(self, index):
        """Get an item of the dataset according to the index.

        :param index: index
        :type index: int
        :return: an item of the dataset according to the index
        :rtype: tensor
        """
        file_name = os.path.join(self.args.root, self.train_list[index])
        data = np.load(file_name)
        rggb = np.transpose(data['arr_0'], (2, 0, 1, 3))
        if self.train:
            target = np.transpose(data['arr_1'], (2, 0, 1, 3))
        else:
            target = data['arr_1']
        ISO = data['arr_2']
        input_ISO = np.ones([1, rggb.shape[1], rggb.shape[2], rggb.shape[3]], np.float) * ISO
        input_rggb = np.vstack((rggb, input_ISO))
        return torch.from_numpy(input_rggb).float(), torch.from_numpy(target).float()
