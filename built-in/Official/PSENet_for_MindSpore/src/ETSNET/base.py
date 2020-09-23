import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal

def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same', has_bias=False):
    init_value = TruncatedNormal(0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=init_value, has_bias=has_bias)

def _bn(channels, momentum=0.1):
    return nn.BatchNorm2d(channels, momentum=momentum)