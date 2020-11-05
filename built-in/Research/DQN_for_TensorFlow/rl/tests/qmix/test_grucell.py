import tensorflow as tf
import numpy as np

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def test():
    batch_size = 10
    depth = 128
    output_dim = 100

    inputs = tf.Variable(tf.random_normal([batch_size, depth]))
    previous_state = tf.Variable(tf.random_normal([batch_size, output_dim]))  # 前一个状态的输出
    gruCell = tf.nn.rnn_cell.GRUCell(output_dim)
    # gruCell = tf.keras.layers.GRUCell(output_dim)

    output, state = gruCell(inputs, previous_state)
    # print(output)
    # print("state, ", state)

    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        # print(sess.run(inputs))
        out = sess.run(output)
        print(np.shape(out))
        sta = sess.run(state)
        print(np.shape(sta))

        print((np.array(out) == np.array(sta)).all())


if __name__ == "__main__":
    test()
