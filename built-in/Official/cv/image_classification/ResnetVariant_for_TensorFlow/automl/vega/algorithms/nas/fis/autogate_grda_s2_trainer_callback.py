# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""AutoGate Grda version Stage2 TrainerCallback."""

import logging
import pandas as pd
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps
from vega.algorithms.nas.fis.ctr_trainer_callback import CtrTrainerCallback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoGateGrdaS2TrainerCallback(CtrTrainerCallback):
    """AutoGateGrdaS2TrainerCallback module."""

    def __init__(self):
        """Construct AutoGateGrdaS2TrainerCallback class."""
        super(CtrTrainerCallback, self).__init__()
        self.sieve_board = pd.DataFrame(
            columns=['selected_feature_pairs', 'score'])
        self.selected_pairs = list()

        logging.info("init autogate s2 trainer callback")

    def before_train(self, logs=None):
        """Call before_train of the managed callbacks."""
        super().before_train(logs)

        """Be called before the training process."""
        hpo_result = FileOps.load_pickle(FileOps.join_path(
            self.trainer.local_output_path, 'best_config.pickle'))
        logging.info("loading stage1_hpo_result \n{}".format(hpo_result))

        self.selected_pairs = hpo_result['feature_interaction']
        print('feature_interaction:', self.selected_pairs)

        model_cfg = ClassFactory.__configs__.get('model')
        # add selected_pairs
        setattr(model_cfg["model_desc"]["custom"], 'selected_pairs', self.selected_pairs)

    def after_train(self, logs=None):
        """Call after_train of the managed callbacks."""
        curr_auc = float(self.trainer.valid_metrics.results['auc'])

        self.sieve_board = self.sieve_board.append(
            {
                'selected_feature_pairs': self.selected_pairs,
                'score': curr_auc
            }, ignore_index=True)
        result_file = FileOps.join_path(
            self.trainer.local_output_path, '{}_result.csv'.format(self.trainer.__worker_id__))

        self.sieve_board.to_csv(result_file, sep='\t')
