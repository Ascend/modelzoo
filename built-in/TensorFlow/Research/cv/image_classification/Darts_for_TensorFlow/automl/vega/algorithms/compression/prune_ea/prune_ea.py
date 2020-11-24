# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Evolution Algorithm used to prune model."""
import logging
import random
import numpy as np
from .conf import PruneConfig
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.report import Report
from vega.search_space.search_algs import SearchAlgorithm


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class PruneEA(SearchAlgorithm):
    """Class of Evolution Algorithm used to Prune Example.

    :param search_space: search space
    :type search_space: SearchSpace
    """

    config = PruneConfig()

    def __init__(self, search_space, **kwargs):
        super(PruneEA, self).__init__(search_space, **kwargs)
        self.length = self.config.policy.length
        self.num_individual = self.config.policy.num_individual
        self.num_generation = self.config.policy.num_generation
        self.random_models = self.config.policy.random_models
        self.random_count = 0
        self.ea_count = 0
        self.ea_epoch = 0

    def crossover(self, ind0, ind1):
        """Cross over operation in EA algorithm.

        :param ind0: individual 0
        :type ind0: list of int
        :param ind1: individual 1
        :type ind1: list of int
        :return: new individual 0, new individual 1
        :rtype: list of int, list of int
        """
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind0.copy()
        b_copy = ind1.copy()
        a_copy[start_idx: end_idx] = ind1[start_idx: end_idx]
        b_copy[start_idx: end_idx] = ind0[start_idx: end_idx]
        return a_copy, b_copy

    def mutatation(self, ind):
        """Mutate operation in EA algorithm.

        :param ind: individual
        :type ind: list of int
        :return: new individual
        :rtype: list of int
        """
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind.copy()
        for k in range(start_idx, end_idx):
            a_copy[k] = 1 - a_copy[k]
        return a_copy

    def search(self):
        """Search one NetworkDesc from search space.

        :return: search id, network desc
        :rtype: int, NetworkDesc
        """
        if self.random_count < self.random_models:
            self.random_count += 1
            desc = self._random_sample()
            desc.update({"trainer.codec": dict(desc)})
            return self.random_count, desc
        self.ea_epoch += 1
        records = Report().get_pareto_front_records(self.step_name, self.num_individual)
        codes = [record.desc.get('backbone').get('encoding') for record in records]
        logging.info("codes=%s", codes)
        if len(codes) < 2:
            encoding1, encoding2 = codes[0], codes[0]
        else:
            encoding1, encoding2 = random.sample(codes, 2)
        choice = random.randint(0, 1)
        # mutate
        if choice == 0:
            encoding_new = self.mutatation(encoding1)
        # crossover
        else:
            encoding_new, _ = self.crossover(encoding1, encoding2)
        self.ea_count += 1
        if self.ea_count % self.num_individual == 0:
            self.ea_epoch += 1
        desc = self.codec.decode(encoding_new)
        desc.update({"trainer.codec": dict(desc)})
        return self.random_count + self.ea_count, desc

    def _random_sample(self):
        """Choose one sample randomly."""
        individual = []
        prob = random.uniform(0, 1)
        for _ in range(self.length):
            s = random.uniform(0, 1)
            if s > prob:
                individual.append(0)
            else:
                individual.append(1)
        return self.codec.decode(individual)

    @property
    def is_completed(self):
        """Whether to complete algorithm."""
        return self.ea_epoch >= self.num_generation
