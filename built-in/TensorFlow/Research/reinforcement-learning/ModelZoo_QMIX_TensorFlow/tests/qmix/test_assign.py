#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test assign variable
from tests.qmix.test_assign import print_mix_tensor_val, print_agent_tensor_val

print_agent_tensor_val(self.graph, self.sess, "before update explore agent")
self._update_explore_agent()

print_agent_tensor_val(self.graph, self.sess, "after update explore agent")

print_mix_tensor_val(self.graph, self.sess, "before update target")
self._update_targets(episode_num=episode_num)
print_mix_tensor_val(self.graph, self.sess, "after update target")
"""

import tensorflow as tf


def print_agent_tensor_val(graph, sess, stage=None):
    if stage:
        print("agent in stage: {}".format(stage))
    explore_k1 = graph.get_tensor_by_name("explore_agent/dense/bias:0")
    target_k1 = graph.get_tensor_by_name("target_agent/dense/bias:0")
    eval_k1 = graph.get_tensor_by_name("eval_agent/dense/bias:0")

    exp_vs_tar = tf.equal(explore_k1, target_k1)
    exp_vs_eval = tf.equal(explore_k1, eval_k1)
    tar_vs_eval = tf.equal(target_k1, eval_k1)
    et, ee, te = sess.run([exp_vs_tar, exp_vs_eval, tar_vs_eval])
    print("exp_vs_tar, exp_vs_eval, tar_vs_eval\n", et, ee, te)

    # ex1, tar1, ev1 = sess.run([explore_k1, target_k1, eval_k1])
    # print("explore_k1, target_k1, eval_k1", ex1, tar1, ev1)


def print_mix_tensor_val(graph, sess, stage=None):
    if stage:
        print("mix in stage: {}".format(stage))
    target_m1 = graph.get_tensor_by_name("target_mixer/hyper_w1/dense/bias:0")
    eval_m1 = graph.get_tensor_by_name("eval_mixer/hyper_w1/dense/bias:0")

    tar_vs_eval = tf.equal(target_m1, eval_m1)

    te = sess.run(tar_vs_eval)
    print("tar_vs_eval\n", te)

    # tar_m1, ev_m1 = sess.run([target_m1, eval_m1])
    # print("target_m1, eval_m1", tar_m1, ev_m1)
