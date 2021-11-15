import tensorflow as tf

from .base_model import VSR
from ascendcv.layers import NormLayer, Conv3D, ActLayer
from ..layers.misc import DynFilter3D, depth_to_space_3D


class DUF(VSR):

    def FR_52L(self, x, norm_cfg=dict(type='bn'), act_cfg=dict(type='ReLU')):
        stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
        sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

        x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), 64, [1, 3, 3], padding='valid', name='conv1')

        F = 64
        G = 16
        for r in range(0, 21):
            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'a')(x)
            t = ActLayer(act_cfg)(t)
            t = Conv3D(t, F, [1, 1, 1], padding='valid', name='Rconv' + str(r + 1) + 'a')

            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'b')(t)
            t = ActLayer(act_cfg)(t)
            t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), G, [3, 3, 3], padding='valid', name='Rconv' + str(r + 1) + 'b')

            x = tf.concat([x, t], 4)
            F += G

        for r in range(21, 24):
            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'a')(x)
            t = ActLayer(act_cfg)(t)
            t = Conv3D(t, F, [1, 1, 1], padding='valid', name='Rconv' + str(r + 1) + 'a')

            t = NormLayer(norm_cfg, is_train=self.is_train, name='Rbn' + str(r + 1) + 'b')(t)
            t = ActLayer(act_cfg)(t)
            t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), G, [3, 3, 3], padding='valid', name='Rconv' + str(r + 1) + 'b')

            x = tf.concat([x[:, 1:-1], t], 4)
            F += G

        x = NormLayer(norm_cfg, is_train=self.is_train, name='fbn1')(x)
        x = ActLayer(act_cfg)(x)
        x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), 256, [1, 3, 3], padding='valid', name='conv2')

        x = ActLayer(act_cfg)(x)

        r = Conv3D(x, 256, [1, 1, 1], padding='valid', name='rconv1')
        r = ActLayer(act_cfg)(r)
        r = Conv3D(r, 3 * 16, [1, 1, 1], padding='valid', name='rconv2')

        f = Conv3D(x, 512, [1, 1, 1], padding='valid', name='fconv1')
        f = ActLayer(act_cfg)(f)
        f = Conv3D(f, 1 * 5 * 5 * 16, [1, 1, 1], padding='valid', name='fconv2')

        ds_f = tf.shape(f)
        f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
        f = tf.nn.softmax(f, dim=4)

        return f, r

    def build_generator(self, x):
        # shape of x: [B,T_in,H,W,C]

        # Generate filters and residual
        # Fx: [B,1,H,W,1*5*5,R*R]
        # Rx: [B,1,H,W,3*R*R]
        with tf.variable_scope('G') as scope:
            Fx, Rx = self.FR_52L(x)

            x_c = []
            for c in range(3):
                t = DynFilter3D(x[:, self.num_frames // 2:self.num_frames // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :],
                                [1, 5, 5])  # [B,H,W,R*R]
                t = tf.depth_to_space(t, self.scale)  # [B,H*R,W*R,1]
                x_c += [t]
            x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3]
            x = tf.expand_dims(x, axis=1)

            Rx = depth_to_space_3D(Rx, self.scale)  # [B,1,H*R,W*R,3]
            x += Rx

        return x[:, 0]

    def caculate_loss(self, SR, HR, delta=0.001, reduction='mean', axis=None):
        abs_error = tf.abs(SR - HR)
        quadratic = tf.minimum(abs_error, delta)
        # The following expression is the same in value as
        # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
        # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
        # This is necessary to avoid doubling the gradient, since there is already a
        # nonzero contribution to the gradient from the quadratic term.
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        if reduction == 'mean':
            return tf.reduce_mean(losses, axis=axis)
        elif reduction == 'sum':
            return tf.reduce_sum(losses, axis=axis)
        else:
            raise NotImplementedError
