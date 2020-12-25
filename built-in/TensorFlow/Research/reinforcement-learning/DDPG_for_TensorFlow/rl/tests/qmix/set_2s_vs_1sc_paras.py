# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
default parameters for 2s_vs_2sc
"""
import tensorflow as tf
from attrdict import AttrDict


def get_args():
    return AttrDict(
        action_selector="epsilon_greedy",
        agent="rnn",
        agent_output_type="q",
        batch_size=32,
        batch_size_run=1,
        buffer_cpu_only=True,
        buffer_size=5000,
        checkpoint_path="",
        critic_lr=0.0005,
        device="cpu",
        double_q=True,  # origin true, set False with tf
        env="sc2",
        env_args={
            "continuing_episode": False,
            "difficulty": "7",
            "game_version": None,
            "map_name": "2s_vs_1sc",
            "move_amount": 2,
            "obs_all_health": True,
            "obs_instead_of_state": False,
            "obs_last_action": False,
            "obs_own_health": True,
            "obs_pathing_grid": False,
            "obs_terrain_height": False,
            "obs_timestep_number": False,
            "reward_death_value": 10,
            "reward_defeat": 0,
            "reward_negative_scale": 0.5,
            "reward_only_positive": True,
            "reward_scale": True,
            "reward_scale_rate": 20,
            "reward_sparse": False,
            "reward_win": 200,
            "replay_dir": "",
            "replay_prefix": "",
            "state_last_action": True,
            "state_timestep_number": False,
            "step_mul": 8,
            "seed": 720353393,
            "heuristic_ai": False,
            "heuristic_rest": False,
            "debug": False,
        },
        epsilon_anneal_time=50000,
        epsilon_finish=0.05,
        epsilon_start=1.0,
        evaluate=False,
        gamma=0.99,
        grad_norm_clip=10,
        hypernet_embed=64,
        hypernet_layers=2,
        label="default_label",
        learner="q_learner",
        learner_log_interval=10000,
        load_step=0,
        local_results_path="results",
        log_interval=10000,
        lr=0.0005,
        mac="basic_mac",
        mixer="qmix",
        mixing_embed_dim=32,
        n_actions=7,
        n_agents=2,
        name="qmix",
        obs_agent_id=True,
        obs_last_action=True,
        optim_alpha=0.99,
        optim_eps=1e-05,
        repeat_id=1,
        rnn_hidden_dim=64,
        runner="episode",
        runner_log_interval=10000,
        save_model=True,
        save_model_interval=2000000,
        save_replay=False,
        seed=720353393,
        state_shape=27,
        t_max=2050000,
        target_update_interval=200,
        test_greedy=True,
        test_interval=10000,
        test_nepisode=32,
        unique_token="qmix__2020-05-12_11-00-45",
        use_cuda=False,
        use_tensorboard=False,
    )


def get_scheme():
    return {
        "state": {"vshape": 27},
        "obs": {"vshape": 17, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": tf.int64},
        "avail_actions": {"vshape": (7,), "group": "agents", "dtype": tf.int32},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": tf.uint8},
        "actions_onehot": {"vshape": (7,), "dtype": tf.float32, "group": "agents"},
        "filled": {"vshape": (1,), "dtype": tf.int64},
    }
