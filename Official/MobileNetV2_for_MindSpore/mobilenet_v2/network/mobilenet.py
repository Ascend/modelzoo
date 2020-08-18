import mindspore.nn as nn
from collections import OrderedDict
from mindspore.ops import operations as P
from mindspore.ops.operations import TensorAdd
from mindspore.common import dtype as mstype
from mindspore import Parameter

from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal
from mobilenet_v2.utils.var_init import KaimingNormal
 

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
 

class GlobalAvgPooling(nn.Cell):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
    
    def construct(self, x):
        x = self.mean(x, (2, 3)) 
        b, c, _, _ = self.shape(x)
        x = self.reshape(x, (b, c))
        return x


class DepthWiseConv(nn.Cell):
    def __init__(self, in_planes, kernel_size, stride, pad_mode, pad, channel_multiplier=1, has_bias=False):
        super(DepthWiseConv, self).__init__()
        self.has_bias = has_bias
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=channel_multiplier, kernel_size=kernel_size, 
                                                      stride=stride, pad_mode=pad_mode, pad=pad)
        self.bias_add = P.BiasAdd()

        weight_shape = [channel_multiplier, in_planes, kernel_size, kernel_size]
        self.weight = Parameter(initializer('ones', weight_shape), name='weight')

        if has_bias:
            bias_shape = [channel_multiplier * in_planes]
            self.bias = Parameter(initializer('zeros', bias_shape), name='bias')
        else:
            self.bias = None
 
    def construct(self, x):
        output = self.depthwise_conv(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class ConvBNReLU(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        if groups == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode="pad", padding=padding, has_bias=False)
        else:
            conv = DepthWiseConv(in_planes, kernel_size, stride, pad_mode="pad", pad=padding)
        
        layers = [conv, nn.BatchNorm2d(out_planes), nn.ReLU6()]
        # layers = [conv, nn.ReLU6()]
        self.features = nn.SequentialCell(layers)
 
    def construct(self, x):
        x = self.features(x)
        return x
 
 
class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1,2]
 
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp== oup
 
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(oup)
        ])
 
        self.conv = nn.SequentialCell(layers)
        self.add = TensorAdd()
        self.cast = P.Cast()
 
    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)
        else:
            return x
 
 
class MobileNetV2(nn.Cell):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class, backbone
 
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
 
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
 
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
 
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.CellList
        self.features = nn.SequentialCell(features)
        self.out_channels = self.last_channel
 
    def construct(self, x):
        x = self.features(x)
        # use mobilenet head
        return x
 
    def get_out_channels(self):
        return self.out_channels


class MobilenetHead(nn.Cell):
    def __init__(self, num_classes, out_channels):
        super(MobilenetHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = nn.Dense(out_channels, num_classes, has_bias=True).add_flags_recursive(fp16=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class Mobilenet_v2(nn.Cell):
    def __init__(self, num_classes):
        super(Mobilenet_v2, self).__init__()
        backbone = MobileNetV2()
        out_channels = backbone.get_out_channels()
        self.head = MobilenetHead(num_classes, out_channels)
        self.backbone = backbone
        self.init_weight()

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def init_weight(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = init.initializer(KaimingNormal(mode='fan_out'), 
                                                             cell.weight.default_input.shape(),
                                                             cell.weight.default_input.dtype()).to_tensor()
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer('zeros', cell.bias.default_input.shape(),
                                                               cell.bias.default_input.dtype()).to_tensor()
            elif isinstance(cell, DepthWiseConv):
                cell.weight.default_input = init.initializer(KaimingNormal(mode='fan_in'), 
                                                             cell.weight.default_input.shape(),
                                                             cell.weight.default_input.dtype()).to_tensor()
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer('zeros', cell.bias.default_input.shape(),
                                                               cell.bias.default_input.dtype()).to_tensor()
            elif isinstance(cell, nn.BatchNorm2d) or isinstance(cell, nn.BatchNorm1d):
                pass
            elif isinstance(cell, nn.Dense):
                cell.weight.default_input = init.initializer(Normal(0.01),
                                                             cell.weight.default_input.shape(),
                                                             cell.weight.default_input.dtype()).to_tensor()
                if cell.bias is not None:
                    cell.bias.default_input = init.initializer("zeros", cell.bias.default_input.shape(),
                                                               cell.bias.default_input.dtype()).to_tensor()


def get_network(backbone_name, num_classes):
    try:
        if backbone_name in ['mobilenet_v2']:
            return Mobilenet_v2(num_classes)
    except:
        raise NotImplementedError('not implement {}'.format(backbone_name))