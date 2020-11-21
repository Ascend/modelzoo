"""
@Author: Jack Qian
@license : Copyright(C), Huawei
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from xt.framework.register import Registers
from xt.model import XTModel
from xt.model.gail.default_config import ENTROPY_LOSS, HIDDEN_UNITS, LR
from xt.model.tf_compat import (
    Adam,
    Conv2D,
    Dense,
    Flatten,
    Input,
    K,
    Lambda,
    Model,
    concatenate,
)
from xt.util.common import import_config


@Registers.model.register
class GailNetworkCnn(XTModel):
    """model for ImpalaNetworkCnn"""

    def __init__(self, model_info):
        model_config = model_info.get("model_config", None)
        import_config(globals(), model_config)

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]
        super(GailNetworkCnn, self).__init__(model_info)
        self.sess.run(tf.initialize_all_variables())

    def create_model(self, model_info):
        self.policy_state = Input(shape=self.state_dim[0], name="policy_state")
        self.policy_action = Input(shape=self.state_dim[1], name="policy_action")
        self.expert_state = Input(shape=self.state_dim[0], name="expert_state")
        self.expert_action = Input(shape=self.state_dim[1], name="expert_action")

        self.feature_model = self.build_graph()
        policy_logits = self.feature_model([self.policy_state, self.policy_action])
        expert_logits = self.feature_model([self.expert_state, self.expert_action])

        # policy_logits = Lambda(noop_func)(policy_logits)
        # expert_logits = Lambda(noop_func)(expert_logits)
        output = concatenate([policy_logits, expert_logits], axis=-1, name="output")

        model = Model(
            inputs=[
                self.policy_state,
                self.policy_action,
                self.expert_state,
                self.expert_action,
            ],
            outputs=output,
        )
        # losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}
        # lossweights = {"output_actions": 1.0, "output_value": .25}
        self.reward = -tf.log(1 - tf.nn.sigmoid(policy_logits) + 1e-8)

        model.compile(optimizer=Adam(lr=LR), loss=gail_loss())
        return model

    def build_graph(self):
        """ build graph """
        state = Input(shape=self.state_dim[0], name="state")
        action = Input(shape=self.state_dim[1], name="action")
        state_input_1 = Lambda(layer_function)(state)
        convlayer = Conv2D(
            32, (8, 8), strides=(4, 4), activation="relu", padding="same"
        )(state_input_1)
        convlayer = Conv2D(
            64, (4, 4), strides=(2, 2), activation="relu", padding="same"
        )(convlayer)
        convlayer = Conv2D(
            64, (3, 3), strides=(1, 1), activation="relu", padding="same"
        )(convlayer)
        denselayer0 = Dense(HIDDEN_UNITS, activation="linear")(action)
        flattenlayer = Flatten()(convlayer)
        denselayer1 = concatenate([flattenlayer, denselayer0], axis=-1)
        logits = Dense(1, activation=tf.identity)(denselayer1)
        model = Model(inputs=[state, action], outputs=logits)
        return model

    def train(self, state, label):
        with self.graph.as_default():
            # print(type(state[2][0][0]))
            K.set_session(self.sess)
            loss = self.model.fit(
                x={
                    "policy_state": state[0],
                    "policy_action": state[1],
                    "expert_state": state[2],
                    "expert_action": state[3],
                },
                y={"output": label,},
                batch_size=128,
                verbose=0,
            )
            return loss

    def predict(self, state):
        """get the output for the state"""
        with self.graph.as_default():
            K.set_session(self.sess)
            return self.sess.run(
                self.reward,
                feed_dict={
                    self.policy_state: state[0],
                    self.policy_action: state[1],
                    self.expert_state: state[2],
                    self.expert_action: state[3],
                },
            )


def layer_function(x):
    """ normalize data """
    return K.cast(x, dtype="float32") / 255.0


def logsigmoid(x):
    """ calc logsigmoid """
    return -tf.nn.softplus(-x)


def logit_bernoulli_entropy(logits):
    """ calc logsit bernoulli entropy """
    ent = (1.0 - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


def gail_loss():
    """loss for impala"""

    def loss(y_true, y_pred):
        policy_logits = y_pred[0]
        expert_logits = y_pred[1]

        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=policy_logits, labels=tf.zeros_like(policy_logits)
        )
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=expert_logits, labels=tf.ones_like(expert_logits)
        )
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([policy_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -ENTROPY_LOSS * entropy
        return generator_loss + expert_loss + entropy_loss

    return loss
