# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined BackboneNas."""
import random
import numpy as np
import logging
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.search_space.search_algs.random_search import RandomSearchAlgorithm
from vega.search_space.search_algs.pareto_front import ParetoFront
from vega.core.common.class_factory import ClassFactory, ClassType
from .conf import BackboneNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BackboneNas(SearchAlgorithm):
    """BackboneNas.

    :param search_space: input search_space
    :type: SeachSpace
    """

    config = BackboneNasConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNas."""
        super(BackboneNas, self).__init__(search_space, **kwargs)
        # ea or random
        self.num_mutate = self.config.policy.num_mutate
        self.random_ratio = self.config.policy.random_ratio
        self.max_sample = self.config.range.max_sample
        self.min_sample = self.config.range.min_sample
        self.sample_count = 0
        logging.info("inited BackboneNas")
        self.pareto_front = ParetoFront(
            self.config.pareto.object_count, self.config.pareto.max_object_ids)
        self.random_search = RandomSearchAlgorithm(self.search_space)
        self._best_desc_file = 'nas_model_desc.json'

    @property
    def is_completed(self):
        """Check if NAS is finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search in search_space and return a sample."""
        sample = {}
        while sample is None or 'code' not in sample:
            pareto_dict = self.pareto_front.get_pareto_front()
            pareto_list = list(pareto_dict.values())
            if self.pareto_front.size < self.min_sample or random.random() < self.random_ratio or len(
                    pareto_list) == 0:
                sample_desc = self.random_search.search()
                sample = self.codec.encode(sample_desc)
            else:
                sample = pareto_list[0]
            if sample is not None and 'code' in sample:
                code = sample['code']
                code = self.ea_sample(code)
                sample['code'] = code
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample):
                sample = None
        self.sample_count += 1
        logging.info(sample)
        sample_desc = self.codec.decode(sample)
        return dict(worker_id=self.sample_count, desc=sample_desc)

    def random_sample(self):
        """Random sample from search_space."""
        sample_desc = self.random_search.search()
        sample = self.codec.encode(sample_desc, is_random=True)
        return sample

    def ea_sample(self, code):
        """Use EA op to change a arch code.

        :param code: list of code for arch
        :type code: list
        :return: changed code
        :rtype: list
        """
        new_arch = code.copy()
        self._insert(new_arch)
        self._remove(new_arch)
        self._swap(new_arch[0], self.num_mutate // 2)
        self._swap(new_arch[1], self.num_mutate // 2)
        return new_arch

    def update(self, record):
        """Use train and evaluate result to update algorithm.

        :param performance: performance value from trainer or evaluator
        """
        perf = record.get("rewards")
        worker_id = record.get("worker_id")
        logging.info("update performance={}".format(perf))
        self.pareto_front.add_pareto_score(worker_id, perf)

    def _insert(self, arch):
        """Random insert to arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        idx = np.random.randint(low=0, high=len(arch[0]))
        arch[0].insert(idx, 1)
        idx = np.random.randint(low=0, high=len(arch[1]))
        arch[1].insert(idx, 1)
        return arch

    def _remove(self, arch):
        """Random remove one from arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        # random pop arch[0]
        ones_index = [i for i, char in enumerate(arch[0]) if char == 1]
        idx = random.choice(ones_index)
        arch[0].pop(idx)
        # random pop arch[1]
        ones_index = [i for i, char in enumerate(arch[1]) if char == 1]
        idx = random.choice(ones_index)
        arch[1].pop(idx)
        return arch

    def _swap(self, arch, R):
        """Random swap one in arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        while True:
            not_ones_index = [i for i, char in enumerate(arch) if char != 1]
            idx = random.choice(not_ones_index)
            r = random.randint(1, R)
            direction = -r if random.random() > 0.5 else r
            try:
                arch[idx], arch[idx + direction] = arch[idx + direction], arch[
                    idx]
                break
            except Exception:
                continue
        return arch
