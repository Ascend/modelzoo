import tensorflow as tf


def gpu_loss_scale_optimizer(opt, loss_scale):
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    if loss_scale == 'off':
        pass
    else:
        if loss_scale.startswith('d'):
            loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
                init_loss_scale=2 ** 32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        elif loss_scale.startswith('f'):
            scale_factor = int(loss_scale[2:])
            loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(
                loss_scale=2 ** scale_factor)
        else:
            raise ValueError
        opt = tf.contrib.mixed_precision.LossScaleOptimizer(opt, loss_scale_manager)

    return opt


def npu_loss_scale_optimizer(opt, loss_scale, is_distributed=False):
    from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
    from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
    from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
    if loss_scale == 'off':
        pass
    else:
        if loss_scale.startswith('d'):
            loss_scale_manager = \
                ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                  decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        elif loss_scale.startswith('f'):
            scale_factor = int(loss_scale[2:])
            loss_scale_manager = FixedLossScaleManager(loss_scale=2 ** scale_factor)
        else:
            raise ValueError
        opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=is_distributed)

    return opt
