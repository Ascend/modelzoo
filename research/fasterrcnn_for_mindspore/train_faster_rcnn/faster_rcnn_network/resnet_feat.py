"""ResNetFea for fasterRCNN."""

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import context
from mindspore.ops import functional as F
from mindspore import context


context.set_context(mode=context.GRAPH_MODE, device_target="Davinci", save_graphs=True)


def weight_init_ones(shape):
    return Tensor(np.array(np.ones(shape).astype(np.float32) * 0.01).astype(np.float16))


def _conv(in_channels, out_channels, kernel_size = 3, stride=1, padding=0, pad_mode='pad'):
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = weight_init_ones(shape)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=False)


def _BatchNorm2dInit(out_chls, momentum=0.1, affine=True, use_batch_statistics=True):
   gamma_init = Tensor(np.array(np.ones(out_chls)).astype(np.float16)) 
   beta_init = Tensor(np.array(np.ones(out_chls) * 0).astype(np.float16)) 
   moving_mean_init = Tensor(np.array(np.ones(out_chls) * 0).astype(np.float16)) 
   moving_var_init = Tensor(np.array(np.ones(out_chls)).astype(np.float16)) 

   return nn.BatchNorm2d(out_chls, momentum=momentum, affine=affine, gamma_init = gamma_init,
                         beta_init = beta_init, moving_mean_init = moving_mean_init,
                         moving_var_init = moving_var_init, use_batch_statistics=use_batch_statistics)


class ResNetFea(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 weights_update = True):
        super(ResNetFea, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        bn_training = False
        self.conv1 = _conv(3, 64, kernel_size = 7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = _BatchNorm2dInit(64, affine=bn_training, use_batch_statistics=bn_training)
        self.relu = P.ReLU()
        self.maxpool = P.MaxPool(ksize=3, strides=2, padding="SAME")
        self.weights_update = weights_update

        if self.weights_update:
            self.conv1.weight.requires_grad = False
        
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=1,
                                       training=bn_training,
                                       weights_update = self.weights_update)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=2,
                                       training=bn_training,
                                       weights_update = False)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=2,
                                       training=bn_training,
                                       weights_update = False)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=2,
                                       training=bn_training,
                                       weights_update = False)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, training = False, weights_update = False):
        layers = []
        down_sample = False
        if stride != 1 or in_channel != out_channel:
            down_sample = True
        resblk = block(in_channel,
                       out_channel,
                       stride=stride,
                       down_sample=down_sample,
                       training = training,
                       weights_update = weights_update)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channel, out_channel, stride=1, training = training, weights_update = weights_update)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        identity = c2
        if self.weights_update:
            identity = F.stop_gradient(c2)
        c3 = self.layer2(identity)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return identity, c3, c4, c5


class ResidualBlockTorch(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channels (int) - Input channel.
        out_channels (int) - Output channel.
        stride (int) - Stride size for the initial convolutional layer. Default: 1.
        down_sample (bool) - If to do the downsample in block. Default: False.
        momentum (float) - Momentum for batchnorm layer. Default: 0.1.

    Returns:
        Tensor, output tensor.

    Examples:
        ResidualBlock(3,256,stride=2,down_sample=True)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False,
                 momentum=0.1,
                 training = False,
                 weights_update = False):
        super(ResidualBlockTorch, self).__init__()

        self.affine = (weights_update==False)

        out_chls = out_channels // self.expansion
        self.conv1 = _conv(in_channels, out_chls, kernel_size = 1, stride=1, padding=0)
        self.bn1 = _BatchNorm2dInit(out_chls, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        self.conv2 = _conv(out_chls, out_chls, kernel_size = 3, stride=stride, padding=1)
        self.bn2 = _BatchNorm2dInit(out_chls, momentum=momentum, affine=self.affine, use_batch_statistics=training)

        self.conv3 = _conv(out_chls, out_channels, kernel_size = 1, stride=1, padding=0)
        self.bn3 = _BatchNorm2dInit(out_channels, momentum=momentum, affine=self.affine, use_batch_statistics=training) 

        if training:
            self.bn1 = self.bn1.set_train()
            self.bn2 = self.bn2.set_train()
            self.bn3 = self.bn3.set_train()

        if weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False

        self.relu = P.ReLU()
        self.downsample = down_sample
        if self.downsample:
            self.conv_down_sample = _conv(in_channels, out_channels, kernel_size = 1, stride=stride, padding=0)
            self.bn_down_sample = _BatchNorm2dInit(out_channels, momentum=momentum, affine=self.affine, use_batch_statistics=training)
            if training:
                self.bn_down_sample = self.bn_down_sample.set_train()
            if weights_update:
                self.conv_down_sample.weight.requires_grad = False
        self.add = P.TensorAdd()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out
