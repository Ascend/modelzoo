# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""AutoGate top-k version Stage2 TrainerCallback."""

import logging
import pandas as pd
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import FileOps
from vega.algorithms.nas.fis.ctr_trainer_callback import CtrTrainerCallback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoGateS2TrainerCallback(CtrTrainerCallback):
    """AutoGateS2TrainerCallback module."""

    def __init__(self):
        """Construct AutoGateS2TrainerCallback class."""
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

        feature_interaction_score = hpo_result['feature_interaction_score']
        print('feature_interaction_score:', feature_interaction_score)
        sorted_pairs = sorted(feature_interaction_score.items(),
                              key=lambda x: abs(x[1]), reverse=True)

        model_cfg = ClassFactory.__configs__.get('model')
        if model_cfg:
            fis_ratio = model_cfg["model_desc"]["custom"]["fis_ratio"]
        else:
            fis_ratio = 1.0
        top_k = int(len(feature_interaction_score) * min(1.0, fis_ratio))
        self.selected_pairs = list(map(lambda x: x[0], sorted_pairs[:top_k]))

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
