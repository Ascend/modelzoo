# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Avazu dataset."""
import numpy as np
import logging
from .common.avazu_util import AVAZUDataset
from .common.dataset import Dataset
from vega.core.common.file_ops import FileOps
from vega.datasets.conf.avazu import AvazuConfig
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.DATASET)
class AvazuDataset(Dataset):
    """This is a class for Avazu dataset.

    :param train: if the mode is train or not, defaults to True
    :type train: bool
    :param cfg: the config the dataset need, defaults to None, and if the cfg is None,
    the default config will be used, the default config file is a yml file with the same name of the class
    :type cfg: yml, py or dict
    """

    config = AvazuConfig()

    def __init__(self, **kwargs):
        """Construct the AvazuDataset class."""
        Dataset.__init__(self, **kwargs)
        self.args.data_path = FileOps.download_dataset(self.args.data_path)
        logging.info("init new avazu_dataset finish. 0721 debug.")

    @property
    def dataloader(self):
        """Dataloader arrtribute which is a unified interface to generate the data.

        :return: a batch data
        :rtype: dict, list, optional
        """
        return AvazuLoader(args=self.args,
                           gen_type=self.mode,
                           batch_size=self.args.batch_size,
                           random_sample=self.args.random_sample,
                           shuffle_block=self.args.shuffle_block,
                           dir_path=self.args.data_path)


class AvazuLoader(AVAZUDataset):
    """Avazu dataset's data loader."""

    def __init__(self, args=None, gen_type="train", batch_size=2000, random_sample=False,
                 shuffle_block=False, dir_path="./"):
        """Construct avazu_loader class."""
        self.args = args
        AVAZUDataset.__init__(self, dir_path=dir_path)
        self.gen_type = gen_type
        self.batch_size = batch_size
        self.random_sample = random_sample
        self.shuffle_block = shuffle_block

    def __iter__(self):
        """Iterate method for AvazuLoader."""
        return self.batch_generator(gen_type=self.gen_type,
                                    batch_size=self.batch_size,
                                    random_sample=self.random_sample,
                                    shuffle_block=self.shuffle_block)

    def __len__(self):
        """Calculate the length of avazu dataset, thus, number of batch."""
        if self.gen_type == "train":
            return int(np.ceil(1.0 * self.args.train_size / self.args.batch_size))
        else:
            return int(np.ceil(1.0 * self.args.test_size / self.args.batch_size))
