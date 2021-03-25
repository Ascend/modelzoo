#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
test train
"""

import tensorflow as tf
import numpy as np
from time import time, strftime, localtime
from xt.algorithm.qmix.qmix import QMixAlgorithm, QMixAgent
from xt.benchmark.visualize import BenchmarkBoard
from tests.qmix.set_2s_vs_1sc_paras import get_args, get_scheme
# from tests.qmix.set_2s3z_paras import get_args, get_scheme
import unittest
from xt.algorithm.qmix.transforms import OneHotNp
import os
from xt.algorithm.qmix.episode_buffer_np import ReplayBufferNP
from absl import logging

NUM_PARALLEL_EXEC_UNITS = 4
os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
# os.environ["KMP_AFFINITY"] = "verbose" # no affinity
# os.environ["KMP_AFFINITY"] = "none" # no affinity
os.environ["KMP_AFFINITY"] = "disabled"  # completely disable thread pools
logging.set_verbosity(logging.INFO)


class TestQMix(unittest.TestCase):
    def setUp(self) -> None:
        self.args = get_args()
        # mac use buffer.scheme, runner used scheme.
        # buffer.scheme
        self.scheme = get_scheme()
        self.agent = QMixAgent(args=self.args, scheme=self.scheme)

    def _init_mix(self):
        args = self.args
        env_info = self.agent.env.get_env_info()
        groups = {"agents": args.n_agents}

        preprocess_np = {
            "actions": ("actions_onehot", [OneHotNp(out_dim=args.n_actions)])
        }
        scheme_np = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": np.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": np.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": np.uint8},
        }

        self.buffer_np = ReplayBufferNP(
            scheme_np,
            groups,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess_np,
        )

        self.agent.setup(scheme=scheme_np, groups=groups, preprocess=preprocess_np)
        print("after setup, scheme: \n{}".format(scheme_np))
        # result = list()
        self.t_env = 0

    def _run_episode(self, episode_times=1):
        for _t in range(episode_times):
            batch = self.agent.run_one_episode(test_mode=False)
            # result.append(batch)
            self.buffer_np.insert_episode_batch(batch)
            self.t_env += self.agent.t

            # for k in (batch.scheme.keys()):
            #     print(k, "*"*20)
            #     print(batch[k])
        # return result

    @unittest.skip
    def test_run_train_explore_same_times(self):
        self._init_mix()
        train_times = 10
        episode_time = self.args.batch_size
        self._run_episode(episode_times=episode_time)
        print("buffer: \n", self.buffer_np)

        if self.buffer_np.can_sample(self.args.batch_size):
            episode_sample = self.buffer_np.sample(self.args.batch_size)
            for _train_times in range(train_times):
                print("start train time-{}".format(_train_times))
                self.agent.train(
                    episode_sample, t_env=self.t_env, episode_num=episode_time
                )

            print("train end")
        else:
            print("NOT Train without sample data!")

    @unittest.skip
    def test_run_train_explore_double_times(self):
        self._init_mix()
        train_times = 10
        episode_time = self.args.batch_size * 2

        self._run_episode(episode_times=episode_time)
        print("buffer: \n", self.buffer_np)

        if self.buffer_np.can_sample(self.args.batch_size):
            episode_sample = self.buffer_np.sample(self.args.batch_size)
            for _train_times in range(train_times):
                print("start train time-{}".format(_train_times))
                self.agent.train(
                    episode_sample, t_env=self.t_env, episode_num=_train_times
                )

            print("train end")
        else:
            print("NOT Train without sample data!")

    # @unittest.skip
    def test_train_with_explore(self):
        self._init_mix()
        max_steps = 10000  # 2050000  #
        episode = 0
        from collections import deque

        explore_time, insert_time = deque(maxlen=100), deque(maxlen=100)
        sample_time, train_time = deque(maxlen=100), deque(maxlen=100)

        def show_time(text, time_list, last_n=9):
            print(
                "{} mean: {},  last-{} as: \n {}".format(
                    text, np.mean(time_list), last_n, list(time_list)[-last_n:]
                )
            )
            return np.mean(time_list)

        self.bm_writer = BenchmarkBoard(
            "logdir",
            "qmix_{}_{}".format(
                self.args.env_args["map_name"], strftime("%Y-%m-%d %H-%M-%S", localtime())
            ),
        )

        last_test_step = -9999
        last_record_step, last_record_epi = -9999, 0
        explore_stats = {}

        while self.agent.t_env < max_steps:
            _t = time()
            batch, records, _info = self.agent.run_one_episode(test_mode=False)
            explore_time.append(time() - _t)

            self.t_env += self.agent.t
            episode += 1

            self.bm_writer.insert_records(records)
            explore_stats.update(
                {
                    k: explore_stats.get(k, 0) + _info.get(k, 0)
                    for k in set(explore_stats) | set(_info)
                }
            )
            if self.t_env - last_record_step >= self.args.runner_log_interval:
                self.bm_writer.insert_records(
                    [
                        (
                            "explore_won_mean",
                            explore_stats["battle_won"] / (episode - last_record_epi),
                            self.t_env,
                        )
                    ]
                )

                explore_stats.clear()
                last_record_epi = episode
                last_record_step = self.t_env

            _t = time()
            self.buffer_np.insert_episode_batch(batch)
            insert_time.append(time() - _t)

            if self.buffer_np.can_sample(self.args.batch_size):
                _t = time()
                episode_sample = self.buffer_np.sample(self.args.batch_size)
                sample_time.append(time() - _t)

                _t = time()
                train_records = self.agent.train(
                    episode_sample, t_env=self.t_env, episode_num=episode
                )
                train_time.append(time() - _t)
                self.bm_writer.insert_records(train_records)

            if self.t_env - last_test_step >= self.args.test_interval:
                test_stats = {}
                for _epi in range(self.args.test_nepisode):
                    _, _rt, env_info = self.agent.run_one_episode(test_mode=True)
                    test_stats.update(
                        {
                            k: test_stats.get(k, 0) + env_info.get(k, 0)
                            for k in set(test_stats) | set(env_info)
                        }
                    )
                print(test_stats)
                print(
                    "test_battle_won_mean",
                    test_stats["battle_won"] / self.args.test_nepisode,
                )

                self.bm_writer.insert_records(
                    [
                        (
                            "test_battle_won_mean",
                            test_stats["battle_won"] / self.args.test_nepisode,
                            self.t_env,
                        )
                    ]
                )

                last_test_step = self.t_env

            if episode % 100 == 99:
                show_time("explore: ", explore_time)
                show_time("insert_sample: ", insert_time)
                show_time("sample data: ", sample_time)
                show_time("train: ", train_time)

        print("train end with episode: {}, t_env: {}".format(episode, self.t_env))
