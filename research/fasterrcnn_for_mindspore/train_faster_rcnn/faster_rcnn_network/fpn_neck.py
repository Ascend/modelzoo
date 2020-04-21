'''

Fpn neck features for fasterRCNN

'''

import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer

context.set_context(mode=context.GRAPH_MODE, device_target="Davinci", save_graphs=True)


def weight_init_ones(shape):
    return Tensor(np.array(np.ones(shape).astype(np.float32) * 0.01).astype(np.float16))


def bias_init_ones(shape):
    return Tensor(np.array(np.ones(shape).astype(np.float32) * 0.01).astype(np.float16))


def bias_init_zeros(shape):
    return Tensor(np.array(np.zeros(shape).astype(np.float32)).astype(np.float16))

def _conv(in_channels, out_channels, kernel_size = 3, stride=1, padding=0, pad_mode='pad'):
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = weight_init_ones(shape)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=False)

class FeatPyramidNeck(nn.Cell):
    """
    Feature pyramid network cell, usually uses as network neck.

    Applies the convolution on multiple, input feature maps
    and output feature map with same channel size. if required num of
    output larger then num of inputs, add extra maxpooling for further
    downsampling;

    Args:
        in_channels (tuple) - Channel size of input feature maps.
        out_channels (int) - Channel size output.
        num_outs (int) - Num of output features.

    Returns:
        Tuple, with tensors of same channel size.

    Examples:
        neck = FeatPyramidNeck([100,200,300], 50, 4)
        input_data = (normal(0,0.1,(1,c,1280//(4*2**i), 768//(4*2**i)),
                      dtype=np.float32) \
                      for i, c in enumerate(config.fpn_in_channels))
        x = neck(input_data)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs):
        super(FeatPyramidNeck, self).__init__()

        self.num_outs = num_outs
        self.in_channels = in_channels
        self.fpn_layer = len(self.in_channels)

        assert not self.num_outs < len(in_channels)

        self.lateral_convs_list_ = []
        self.fpn_convs_ = []

        for i in range(len(in_channels)):
            l_conv = _conv(in_channels[i], out_channels, kernel_size=1, stride=1, padding=0, pad_mode='valid')
            fpn_conv = _conv(out_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same')
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.interpolate1 = P.ResizeNearestNeighbor((48, 80))
        self.interpolate2 = P.ResizeNearestNeighbor((96, 160))
        self.interpolate3 = P.ResizeNearestNeighbor((192, 320))
        self.maxpool = P.MaxPool(ksize=1, strides=2, padding="same")

    def construct(self, inputs):
        x = ()
        for i in range(self.fpn_layer):
            x += (self.lateral_convs_list[i](inputs[i]),)

        y = (x[3],)
        y = y + (x[2] + self.interpolate1(y[self.fpn_layer - 4]),)
        y = y + (x[1] + self.interpolate2(y[self.fpn_layer - 3]),)
        y = y + (x[0] + self.interpolate3(y[self.fpn_layer - 2]),)

        z = ()
        for i in range(self.fpn_layer - 1, -1, -1):
            z = z + (y[i],)

        outs = ()
        for i in range(self.fpn_layer):
            outs = outs + (self.fpn_convs_list[i](z[i]),)

        for i in range(self.num_outs - self.fpn_layer):
            outs = outs + (self.maxpool(outs[3]),)
        return outs
