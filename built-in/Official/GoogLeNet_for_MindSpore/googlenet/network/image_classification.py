import mindspore.nn as nn
from mindspore.common import initializer as init
import math

import googlenet.backbones as backbones
import googlenet.network.head as heads
from googlenet.utils.var_init import default_recurisive_init, KaimingNormal


class ImageClassificationNetwork(nn.Cell):
    def __init__(self, backbone, head):
        super(ImageClassificationNetwork, self).__init__()
        self.backbone = backbone
        self.head = head

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class GoogLeNet(ImageClassificationNetwork):
    def __init__(self, backbone_name, num_classes, is_train=True):
        self.backbone_name = backbone_name
        backbone = backbones.__dict__[self.backbone_name]()
        out_channels = backbone.get_out_channels()
        head = heads.GoogLeNetHead(num_classes=num_classes, out_channels=out_channels, is_train=is_train)
        super(GoogLeNet, self).__init__(backbone, head)


def get_network(backbone_name, num_classes, is_train=True):
    try:
        if backbone_name in ['googlenet']:
            return GoogLeNet(backbone_name, num_classes, is_train)
    except:
        raise NotImplementedError('not implement {}'.format(backbone_name))
