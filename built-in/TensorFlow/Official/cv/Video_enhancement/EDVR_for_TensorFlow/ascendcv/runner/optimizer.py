import tensorflow as tf


def build_optimizer(lr, opt_cfg, device, is_distributed, mix_precision, loss_scale):

    opt_type = opt_cfg.type.lower()

    if opt_type == 'adam':
        beta1 = opt_cfg.get('beta1', 0.9)
        beta2 = opt_cfg.get('beta2', 0.999)
        epsilon = opt_cfg.get('epsilon', 1e-08)
        opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    elif opt_type == 'momentum':
        momentum = opt_cfg.get('momentum', 0.9)
        opt = tf.train.MomentumOptimizer(lr, momentum=momentum)
    else:
        raise KeyError('Unkown type {}'.format(opt_type))

    if device == 'npu':
        # if not mix_precision:
        #    raise ValueError('mix precision must be enable on NPU')
        from .loss_scaling import npu_loss_scale_optimizer
        opt = npu_loss_scale_optimizer(opt, loss_scale)

    if device == 'gpu' and mix_precision:
        from .loss_scaling import gpu_loss_scale_optimizer
        opt = gpu_loss_scale_optimizer(opt, loss_scale)
    
    if device == 'npu' and is_distributed:
        from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
        opt = NPUDistributedOptimizer(opt)        

    return opt
