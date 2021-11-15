import tensorflow as tf

from .dufvsr import DUF
from ascendcv.layers import NormLayer, Conv2D, ActLayer


class DUFwP3D(DUF):

    def FR_52L(self, x, norm_cfg=dict(type='bn'), act_cfg=dict(type='ReLU')):
        sp = [[0, 0], [1, 1], [1, 1], [0, 0]]
        tp = [[0, 0], [1, 1], [0, 0], [0, 0]]

        # shape0 = list(map(int, x.shape))
        shape0 = x.get_shape().as_list()
        x = tf.reshape(x, [shape0[0] * shape0[1], shape0[2], shape0[3], shape0[4]])

        x = Conv2D(tf.pad(x, sp, mode='CONSTANT'), 64, [3, 3], padding='valid', name='conv1')

        F = 64
        G = 16
        for r in range(0, 21):
            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'a')(x)
            t = ActLayer(act_cfg)(t)
            t = Conv2D(t, F, [1, 1], padding='valid', name='Rconv' + str(r + 1) + 'a')

            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'b')(t)
            t = ActLayer(act_cfg)(t)
            # P3D
            t = tf.pad(t, sp, mode='CONSTANT')
            t = Conv2D(t, G, [3, 3], padding='valid', name='Rconv' + str(r + 1) + 'b_1x3x3')
            t = ActLayer(act_cfg)(t)
            t = tf.reshape(t, [shape0[0], int(t.shape[0]) // shape0[0], int(t.shape[1]), int(t.shape[2]), int(t.shape[3])])
            t = tf.transpose(t, [0, 2, 1, 3, 4])
            t = tf.reshape(t, [int(t.shape[0]) * int(t.shape[1]), int(t.shape[2]), int(t.shape[3]), int(t.shape[4])])
            t = tf.pad(t, tp, mode='CONSTANT')
            t = Conv2D(t, G, [3, 1], padding='valid', name='Rconv' + str(r + 1) + 'b_3x1x1')
            t = tf.reshape(t, [shape0[0], int(t.shape[0]) // shape0[0], int(t.shape[1]), int(t.shape[2]), int(t.shape[3])])
            t = tf.transpose(t, [0, 2, 1, 3, 4])
            t = tf.reshape(t, [int(t.shape[0]) * int(t.shape[1]), int(t.shape[2]), int(t.shape[3]), int(t.shape[4])])

            x = tf.concat([x, t], 3)
            F += G

        for r in range(21, 24):
            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'a')(x)
            t = ActLayer(act_cfg)(t)
            t = Conv2D(t, F, [1, 1], padding='valid', name='Rconv' + str(r + 1) + 'a')

            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'b')(t)
            t = ActLayer(act_cfg)(t)
            # P3D
            t = tf.pad(t, sp, mode='CONSTANT')
            t = Conv2D(t, G, [3, 3], padding='valid', name='Rconv' + str(r + 1) + 'b_1x3x3')
            t = ActLayer(act_cfg)(t)
            t = tf.reshape(t, [shape0[0], int(t.shape[0]) // shape0[0], int(t.shape[1]), int(t.shape[2]), int(t.shape[3])])
            t = tf.transpose(t, [0, 2, 1, 3, 4])
            t = tf.reshape(t, [int(t.shape[0]) * int(t.shape[1]), int(t.shape[2]), int(t.shape[3]), int(t.shape[4])])
            t = Conv2D(t, G, [3, 1], padding='valid', name='Rconv' + str(r + 1) + 'b_3x1x1')
            t = tf.reshape(t, [shape0[0], int(t.shape[0]) // shape0[0], int(t.shape[1]), int(t.shape[2]), int(t.shape[3])])
            t = tf.transpose(t, [0, 2, 1, 3, 4])
            t = tf.reshape(t, [int(t.shape[0]) * int(t.shape[1]), int(t.shape[2]), int(t.shape[3]), int(t.shape[4])])

            x = tf.reshape(x, [shape0[0], int(x.shape[0]) // shape0[0], int(x.shape[1]), int(x.shape[2]), int(x.shape[3])])
            x = x[:, 1:-1]
            x = tf.reshape(x, [int(x.shape[0]) * int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4])])
            x = tf.concat([x, t], 3)
            F += G

        x = NormLayer(norm_cfg, is_train=self.is_train, name='fbn1')(x)
        x = ActLayer(act_cfg)(x)
        x = Conv2D(tf.pad(x, sp, mode='CONSTANT'), 256, [3, 3], padding='valid', name='conv2')

        x = ActLayer(act_cfg)(x)

        r = Conv2D(x, 256, [1, 1], padding='valid', name='rconv1')
        r = ActLayer(act_cfg)(r)
        r = Conv2D(r, 3 * 16, [1, 1], padding='valid', name='rconv2')
        r = tf.reshape(r, [shape0[0], int(r.shape[0]) // shape0[0], int(r.shape[1]), int(r.shape[2]), int(r.shape[3])])

        f = Conv2D(x, 512, [1, 1], padding='valid', name='fconv1')
        f = ActLayer(act_cfg)(f)
        f = Conv2D(f, 1 * 5 * 5 * 16, [1, 1], padding='valid', name='fconv2')
        f = tf.nn.softmax(f, dim=3)
        f = tf.reshape(f, [shape0[0], int(f.shape[0]) // shape0[0], int(f.shape[1]), int(f.shape[2]), 25, 16])

        return f, r
