import tensorflow as tf

def _fp32_trainvar_getter(getter, name, shape=None, dtype=None,
                          trainable=True, regularizer=None,
                          *args, **kwargs):
    storage_dtype = dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      trainable=trainable,
                      regularizer=regularizer if trainable and 'BatchNorm' not in name and 'batchnorm' not in name and 'batch_norm' not in name and 'Batch_Norm' not in name else None,
                      *args, **kwargs)

    return variable


def fp32_trainable_vars(name='fp32_vars', *args, **kwargs):
    """A varible scope with custom variable getter to convert fp16 trainable
    variables with fp32 storage followed by fp16 cast.
    """
    return tf.variable_scope(
        name, custom_getter=_fp32_trainvar_getter, *args, **kwargs)

def custom_getter_with_fp16_and_weight_decay(dtype, weight_decay):
    return fp32_trainable_vars(dtype=dtype, regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

