# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""MFKD1."""

import copy
import itertools
import numpy as np
import logging
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from vega.core.report import Report
from vega.core.common.utils import update_dict
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.search_algs import SearchAlgorithm


class MFKD1Config(object):
    """Configure of MFKD1."""

    max_samples = 5
    objective_keys = "accuracy"


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class MFKD1(SearchAlgorithm):
    """MFKD1 Search Algorithm."""

    config = MFKD1Config()

    def __init__(self, search_space):
        """Initialize."""
        super(MFKD1, self).__init__(search_space)
        self.max_samples = self.config.max_samples
        self.sample_count = 0
        self.acc_list = []
        self._get_all_arcs()
        self.points = list(np.random.choice(
            range(len(self.X)), size=self.max_samples, replace=False))
        logging.info('Selected %d random points: %s' % (len(self.points), str(self.points)))

    def _sub_config_choice(self, config, choices, pos):
        """Apply choices to config."""
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                _, pos = self._sub_config_choice(value, choices, pos)
            elif isinstance(value, list):
                choice = value[choices[pos]]
                config[key] = choice
                pos += 1
        return config, pos

    def _desc_from_choices(self, choices):
        """Create description object from choices."""
        desc = {}
        pos = 0
        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            module_cfg, pos = self._sub_config_choice(config_space, choices, pos)
            desc[key] = module_cfg
        desc = update_dict(desc, copy.deepcopy(self.search_space))
        return desc

    def _sub_config_all(self, config, vectors, choices):
        """Get all possible choices and their values."""
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                self._sub_config_all(value, vectors, choices)
            elif isinstance(value, list):
                vectors.append([float(x) for x in value])
                choices.append(list(range(len(value))))

    def _get_all_arcs(self):
        """Get all the architectures from the search space."""
        vectors = []
        choices = []
        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            self._sub_config_all(config_space, vectors, choices)
        self.X = list(itertools.product(*vectors))
        self.X = preprocessing.scale(self.X, axis=0)
        self.choices = list(itertools.product(*choices))
        logging.info('Number of architectures in the search space %d' % len(self.X))

    def _get_best_arc(self):
        """Find the best (by estimate) architecture from the search space."""
        X_train = []
        y_train = []
        for i in range(len(self.points)):
            idx = self.points[i]
            X_train.append(self.X[idx])
            y_train.append(self.acc_list[i])
        gpr = GPR(kernel=RBF(1.0))
        gpr.fit(X_train, y_train)
        preds = gpr.predict(self.X, return_std=True)
        best_idx = np.argmax(preds[0])
        return best_idx

    def search(self):
        """Generate a network description and desc's ID."""
        idx = self.points[self.sample_count]
        logging.info('Checking architecture %d' % idx)
        desc = self._desc_from_choices(self.choices[idx])
        self.sample_count += 1
        return self.sample_count, desc

    def update(self, record):
        """Update search algorithm's model after receiving the performance."""
        acc = record.get("rewards")
        self.acc_list.append(acc)

    @property
    def is_completed(self):
        """Check if the search is finished."""
        _completed = self.sample_count >= self.max_samples
        if _completed:
            idx = self._get_best_arc()
            desc = self._desc_from_choices(self.choices[idx])
            self._save_best(desc)
            logging.info('The best architecture %d, description %s' % (idx, str(desc)))
        return _completed

    def _save_best(self, desc):
        record = Report().receive(self.step_name, self.sample_count + 1)
        record.performance = {"accuracy": 100}
        record.desc = desc
        Report().broadcast(record)
