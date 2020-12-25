# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE
"""MCTS module: where MuZero thinks inside the tree."""

import math
import random

import numpy as np

from xt.agent.muzero.default_config import PB_C_BASE, PB_C_INIT
from xt.agent.muzero.default_config import ROOT_DIRICHLET_ALPHA
from xt.agent.muzero.default_config import ROOT_EXPLORATION_FRACTION
from xt.agent.muzero.default_config import GAMMA

from xt.agent.muzero.util import MinMaxStats, Node, soft_max_sample
from xt.model.muzero.muzero_model import NetworkOutput


class Mcts(object):
    """MCTS operation."""
    def __init__(self, agent, root_state):
        self.network = agent.alg.actor
        self.action_dim = agent.alg.action_dim
        self.num_simulations = agent.num_simulations
        self.min_max_stats = MinMaxStats(None)
        self.discount = GAMMA
        self.actions = range(self.action_dim)
        self.pb_c_base = PB_C_BASE
        self.pb_c_init = PB_C_INIT
        self.root_dirichlet_alpha = ROOT_DIRICHLET_ALPHA
        self.root_exploration_fraction = ROOT_EXPLORATION_FRACTION

        self.root = Node(0)
        # policy, value = self.network.ppo_infer(root_state)
        # policy = list(policy[0])
        # value = value[0][0]

        root_state = root_state.reshape((1, ) + root_state.shape)
        network_output = self.network.initial_inference(root_state)
        # print(network_output.policy, policy)
        # new_output = NetworkOutput(network_output.value, network_output.reward, policy, network_output.hidden_state)
        # network_output.policy = policy
        self.init_node(self.root, network_output)
        self.policy = network_output.policy

    def init_node(self, node, network_output):
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward

        # max_policy = max(network_output.policy)
        # policy = [math.exp(p - max_policy) for p in network_output.policy]
        # self.policy = np.array(policy) / sum(policy)

        policy = [p for p in network_output.policy]
        # max_policy = max(network_output.policy[0])
        # policy = [math.exp(p - max_policy) for p in network_output.policy[0]]


        policy_sum = sum(policy)
        for action in self.actions:
            node.children[action] = Node(policy[action] / policy_sum)

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the
        tree to the root.
        """
        for node in search_path[::-1]:
            node.value_sum += value
            node.visit_count += 1
            self.min_max_stats.update(node.value())

            value = node.reward + self.discount * value

    def run_mcts(self):
        """
        Core Monte Carlo Tree Search algorithm.
        To decide on an action, we run N simulations, always starting at the root of
        the search tree and traversing the tree according to the UCB formula until we
        reach a leaf node.
        """
        for _ in range(self.num_simulations):
            node = self.root
            search_path = [node]
            history = []

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                history.append(action)
            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            network_output = self.network.recurrent_inference(parent.hidden_state, history[-1])
            self.init_node(node, network_output)

            self.backpropagate(search_path, network_output.value)

    def select_action(self, mode='softmax'):
        """
        After running simulations inside in MCTS, we select an action based on the root's children visit counts.
        During training we use a softmax sample for exploration.
        During evaluation we select the most visited child.
        """
        node = self.root
        visit_counts = [child.visit_count for child in node.children.values()]
        actions = self.actions
        action = None
        if mode == 'softmax':
            action = soft_max_sample(visit_counts, actions, 1)
        elif mode == 'max':
            action = np.argmax(visit_counts)
        return action

    def ucb_score(self, parent, child):
        """
        The score for a node is based on its value, plus an exploration bonus based on
        the prior.
        """
        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        # value_score = self.min_max_stats.normalize(child.value())
        if child.visit_count > 0:
            # value_score = child.reward + self.discount * self.min_max_stats.normalize(child.value())
            value_score = self.min_max_stats.normalize(child.value())
        else:
            value_score = 0
        return prior_score + value_score

    def add_exploration_noise(self, node):
        actions = self.actions
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * self.action_dim)
        frac = self.root_exploration_fraction
        for i, _noise in zip(actions, noise):
            node.children[i].prior = node.children[i].prior * (1 - frac) + _noise * frac

    def get_info(self):
        """ get train info from mcts tree """
        child_visits = [self.root.children[a].visit_count for a in self.actions]
        sum_visits = sum(child_visits)
        child_visits = [visits / sum_visits for visits in child_visits]
        # child_visits = [1 / 4 for visits in child_visits]
        # print(child_visits, sum_visits)
        # return {"child_visits": child_visits, "root_value": np.asscalar(self.root.value())}
        return {"child_visits": child_visits, "root_value": self.root.value()}

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        _, action, child = max((self.ucb_score(node, child), action, child) for action, child in node.children.items())
        return action, child
