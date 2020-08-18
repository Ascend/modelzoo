import mindspore
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal
import numpy as np

__all__ = ['GoogLeNet', 'googlenet']


def googlenet():
    return GoogLeNet()


class GoogLeNet(nn.Cell):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        conv_block = BasicConv2d
        inception_block = Inception
        self.out_channels = 1024
        self.aux1_ch = 512
        self.aux2_ch = 528

        # N*3*224*224 +++ change to 228*228
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        # N*64*112*112  +++ change to 114*114
        self.mp1 = nn.MaxPool2d(3, stride=2)
        # N*64*55*55
        self.conv2 = conv_block(64, 64)
        # N*64*55*55
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        # N*192*55*55
        self.mp2 = nn.MaxPool2d(3, stride=2)
        # N*192*27*27

        self.icpt_3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        # N*256*27*27
        self.icpt_3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        # N*480*27*27
        self.mp3 = nn.MaxPool2d(3, stride=2)
        # N*480*13*13

        self.icpt_4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        # N*512*13*13
        self.icpt_4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        # N*512*13*13
        self.icpt_4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        # N*512*13*13
        self.icpt_4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        # N*528*13*13
        self.icpt_4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        # N*832*13*13
        self.mp4 = nn.MaxPool2d(3, stride=2)
        # N*832*6*6

        self.icpt_5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        # N*832*6*6
        self.icpt_5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        # N*1024*6*6

    def construct(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mp2(x)
        x = self.icpt_3a(x)
        x = self.icpt_3b(x)
        x = self.mp3(x)
        aux1 = self.icpt_4a(x)
        x = self.icpt_4b(aux1)
        x = self.icpt_4c(x)
        aux2 = self.icpt_4d(x)
        x = self.icpt_4e(aux2)
        x = self.mp4(x)
        x = self.icpt_5a(x)
        x = self.icpt_5b(x)

        # return x, aux1, aux2
        return x

    def get_out_channels(self):
        # return self.out_channels, self.aux1_ch, self.aux2_ch
        return self.out_channels


class Inception(nn.Cell):
    # 192, 64, 96, 128, 16, 32, 32
    # 256, 128, 128, 192, 32, 96, 64
    # 480, 192, 96, 208, 16, 48, 64
    def __init__(self, ch_in, ch_1, ch_31, ch_32, ch_51, ch_52, ch_ot):
        super(Inception, self).__init__()
        self.concat = P.Concat(axis=1)
        self.branch1 = BasicConv2d(ch_in, ch_1)
        self.branch2_1 = BasicConv2d(ch_in, ch_31)
        self.branch2_2 = BasicConv2d(ch_31, ch_32, kernel_size=3, padding=1)
        self.branch3_1 = BasicConv2d(ch_in, ch_51)
        self.branch3_2 = BasicConv2d(ch_51, ch_52, kernel_size=3, padding=1)
        self.branch4_1 = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.branch4_2 = BasicConv2d(ch_in, ch_ot)

    # Layer   input        x1   x2   x3   x4   out   fea_map
    # 3a      N*192*27*27, 64,  128, 32,  32   256   27*27
    # 3b      N*256*27*27, 128, 192, 96,  64   480   27*27
    # 4a      N*480*13*13, 192, 208, 48,  64   512   13*13

    def construct(self, x):
        # N*192*27*27
        x1 = self.branch1(x)
        # N*64*27*27
        x2_1 = self.branch2_1(x)
        # N*96*27*27
        x2_2 = self.branch2_2(x2_1)
        # N*128*27*27
        x3_1 = self.branch3_1(x)
        # N*16*27*27
        x3_2 = self.branch3_2(x3_1)
        # N*32*27*27
        x4_1 = self.branch4_1(x)
        # N*192*27*27
        x4_2 = self.branch4_2(x4_1)
        # N*32*27*27

        out = self.concat((x1, x2_2, x3_2, x4_2))
        # 64 + 128 + 32 + 32 = 256
        return out


class BasicConv2d(nn.Cell):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding="valid"):
        super(BasicConv2d, self).__init__()
        self.conv = conv2DSelfDef(in_planes, out_planes, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)  # gamma 1s, beta 0s, by default
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def conv2DSelfDef(in_planes, out_planes, kernel_size=1, stride=1, padding="valid", bias=True, pad_mode="valid"):
    if padding == "same":
        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=kernel_size, stride=stride, padding=0,
                         has_bias=bias, pad_mode="same", weight_init='HeUniform')

    if padding == "valid":
        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=kernel_size, stride=stride, padding=0,
                         has_bias=bias, pad_mode="valid", weight_init='HeUniform')

    if type(padding) == type((1, 2)):
        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=kernel_size, stride=stride, padding=0,
                         has_bias=bias, pad_mode="same", weight_init='HeUniform')

    if type(padding) == type(2):
        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         has_bias=bias, pad_mode="pad", weight_init='HeUniform')
