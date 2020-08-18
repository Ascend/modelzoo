import mindspore.nn as nn
from mindspore.ops.operations import TensorAdd
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

from resnet50.blocks import SEBlock


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def weight_variable(shape, factor=0.1):
    return TruncatedNormal(0.02)


def conv7x7(in_channels, out_channels, stride=1, padding=3, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv3x3(in_channels, out_channels, stride=1, padding=1, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


def conv1x1(in_channels, out_channels, stride=1, padding=0, has_bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=has_bias,
                     padding=padding, pad_mode="pad")


class _DownSample(nn.Cell):
    def __init__(self, in_channels, out_channels, stride):
        super(_DownSample, self).__init__()
        self.conv = conv1x1(in_channels, out_channels, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
 
    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None, base_width=64, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = P.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels)

        self.down_sample_flag = False
        if down_sample is not None:
            self.down_sample = down_sample
            self.down_sample_flag = True

        self.add = TensorAdd()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.down_sample_flag:
            identity = self.down_sample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None, base_width=64, use_se=False):
        super(Bottleneck, self).__init__()

        width = int(out_channels * (base_width / 64.0))

        self.conv1 = conv1x1(in_channels, width, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = P.ReLU()
        self.conv2 = conv3x3(width, width, stride=stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels * self.expansion)

        self.down_sample_flag = False
        if down_sample is not None:
            self.down_sample = down_sample
            self.down_sample_flag = True

        self.add = TensorAdd()

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

        if self.use_se:
            out = self.se(out)

        if self.down_sample_flag:
            identity = self.down_sample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    def __init__(self, block, layers, width_per_group=64, use_se=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.base_width = width_per_group

        self.conv = conv7x7(3, self.in_channels, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block,  64, layers[0], use_se=use_se)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=use_se)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=use_se)

        self.out_channels = 512 * block.expansion

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, out_channels, blocks_num, stride=1, use_se=False):
        down_sample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            down_sample = _DownSample(self.in_channels,
                                      out_channels * block.expansion,
                                      stride=stride)

        layers = []
        layers.append(block(self.in_channels, 
                            out_channels,
                            stride=stride,
                            down_sample=down_sample,
                            base_width=self.base_width,
                            use_se=use_se))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks_num):
            layers.append(block(self.in_channels, out_channels, base_width=self.base_width, use_se=use_se))

        return nn.SequentialCell(layers)

    def get_out_channels(self):
        return self.out_channels


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

