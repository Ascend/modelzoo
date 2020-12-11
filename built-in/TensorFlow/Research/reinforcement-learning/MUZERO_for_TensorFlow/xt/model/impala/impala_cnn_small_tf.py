"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from xt.model.tf_compat import (
    Conv2D,
    Lambda,
    Flatten,
    K,
    DTYPE_MAP,
    tf,
)

from xt.model.impala.default_config import ENTROPY_LOSS, LR
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model.register
class ImpalaCnnNetTF(XTModel):
    """model for ImpalaNetworkCnn"""

    def __init__(self, model_info):
        model_config = model_info.get("model_config", None)
        import_config(globals(), model_config)

        model_config = model_info.get("model_config", dict())
        import_config(globals(), model_config)
        self.dtype = DTYPE_MAP.get(model_config.get("dtype", "float32"))

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]

        self.ph_state = None
        self.ph_adv = None
        self.out_actions = None
        self.out_val = None

        self.ph_target_action = None
        self.ph_target_val = None
        self.loss, self.optimizer, self.train_op = None, None, None
        self.saver = None

        super(ImpalaCnnNetTF, self).__init__(model_info)

    def create_model(self, model_info):
        self.ph_state = tf.placeholder(
            self.dtype, shape=(None, *self.state_dim,), name="state_input"
        )
        self.ph_adv = tf.placeholder(self.dtype, shape=(None, 1), name="adv")

        self.ph_target_action = tf.placeholder(
            self.dtype, shape=(None, self.action_dim), name="target_action"
        )
        self.ph_target_val = tf.placeholder(
            self.dtype, shape=(None, 1), name="target_value"
        )

        state_input_1 = Lambda(lambda x: K.cast(x, dtype="float32") / 255.0)(
            self.ph_state
        )

        convlayer = Conv2D(
            16, (4, 4), strides=(2, 2), activation="relu", padding="same"
        )(state_input_1)
        convlayer = Conv2D(
            32, (4, 4), strides=(2, 2), activation="relu", padding="same"
        )(convlayer)
        print(convlayer)
        convlayer = Conv2D(
            256, (11, 11), strides=(1, 1), activation="relu", padding="valid"
        )(convlayer)

        policy_conv = Conv2D(
            self.action_dim, (1, 1), strides=(1, 1), activation="relu", padding="valid"
        )(convlayer)
        flattenlayer = Flatten()(policy_conv)
        self.out_actions = tf.layers.dense(
            inputs=flattenlayer, units=self.action_dim, activation=tf.nn.softmax
        )
        flattenlayer = Flatten()(convlayer)

        self.out_val = tf.layers.dense(inputs=flattenlayer, units=1, activation=None)

        self.loss = 0.5 * tf.losses.mean_squared_error(
            self.ph_target_val, self.out_val
        ) + impala_loss(self.ph_adv, self.ph_target_action, self.out_actions)

        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=10)
        return True

    def train(self, state, label, train_batch_size=128):
        """train with sess.run"""
        loss_val = []
        with self.graph.as_default():
            nbatch = state[0].shape[0]
            inds = np.arange(nbatch)

            np.random.shuffle(inds)
            for start in range(0, nbatch, train_batch_size):
                end = start + train_batch_size
                mbinds = inds[start:end]

                _, loss = self.sess.run(
                    [self.train_op, self.loss],
                    feed_dict={
                        self.ph_state: state[0][mbinds],
                        self.ph_adv: state[1][mbinds],
                        self.ph_target_action: label[0][mbinds],
                        self.ph_target_val: label[1][mbinds],
                    },
                )
                loss_val.append(loss)

        return np.mean(loss_val)

    def predict(self, state):
        """
        Do predict use the newest model.
        :param state:
        :return:
        """
        with self.graph.as_default():
            feed_dict = {self.ph_state: state[0]}
            return self.sess.run([self.out_actions, self.out_val], feed_dict)

    def save_model(self, file_name):
        ck_name = self.saver.save(
            self.sess, save_path=file_name, write_meta_graph=False
        )
        # print("save: ", ck_name)
        return ck_name

    def load_model(self, model_name, by_name=False):
        # print(">> load model: {}".format(model_name))
        self.saver.restore(self.sess, model_name)


def impala_loss(advantage, y_true, y_pred):
    """loss for impala"""
    policy = y_pred
    log_policy = K.log(policy + 1e-10)
    entropy = -policy * K.log(policy + 1e-10)
    cross_entropy = -y_true * log_policy
    return K.mean(advantage * cross_entropy - ENTROPY_LOSS * entropy)
